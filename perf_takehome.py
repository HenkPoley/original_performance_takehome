"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        class DependencyAwarePacker:
            def __init__(self, slot_limits):
                self.slot_limits = slot_limits
                self.cycle_instrs = []
                self.last_write = {}
                self.last_read = {}

            def add(self, engine, instr, reads, writes):
                start_cycle = 0
                for r in reads:
                    if r in self.last_write:
                        start_cycle = max(start_cycle, self.last_write[r] + 1)
                for w in writes:
                    if w in self.last_write:
                        start_cycle = max(start_cycle, self.last_write[w] + 1)
                    if w in self.last_read:
                        start_cycle = max(start_cycle, self.last_read[w] + 1)
                c = start_cycle
                while True:
                    if c >= len(self.cycle_instrs): self.cycle_instrs.append({})
                    if engine not in self.cycle_instrs[c]: self.cycle_instrs[c][engine] = []
                    if len(self.cycle_instrs[c][engine]) < self.slot_limits.get(engine, 64):
                        self.cycle_instrs[c][engine].append(instr)
                        for w in writes: self.last_write[w] = c
                        for r in reads: self.last_read[r] = max(self.last_read.get(r, 0), c)
                        break
                    c += 1

            def get_instrs(self):
                return self.cycle_instrs

        # 1. Variables
        init_vars = ["rounds", "n_nodes", "batch_size", "forest_height", "forest_values_p", "inp_indices_p", "inp_values_p"]
        for v in init_vars: self.alloc_scratch(v)
        tmp1 = self.alloc_scratch("tmp1")
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))
        self.add("flow", ("pause",))

        # Vector constants
        v_const_map = {}
        def get_v_const(val):
            if val in v_const_map: return v_const_map[val]
            addr = self.alloc_scratch(None, VLEN)
            t_addr = self.scratch_const(val)
            self.add("valu", ("vbroadcast", addr, t_addr))
            v_const_map[val] = addr
            return addr

        v_zero = get_v_const(0); v_one = get_v_const(1); v_two = get_v_const(2)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
        self.add("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))

        v_mul = [get_v_const(2**s + 1) for s in [12, 5, 3]]
        v_add = [get_v_const(HASH_STAGES[s][1]) for s in [0, 2, 4]]
        v_hash_consts = {}
        for hi in [1, 3, 5]:
            v_hash_consts[(hi, 1)] = get_v_const(HASH_STAGES[hi][1])
            v_hash_consts[(hi, 3)] = get_v_const(HASH_STAGES[hi][4])

        # 2. Cache top levels (0-2)
        v_cached = []
        for i in range(7):
            v_c = self.alloc_scratch(f"v_node_{i}", VLEN)
            t_addr = self.alloc_scratch(None)
            self.add("alu", ("+", t_addr, self.scratch["forest_values_p"], self.scratch_const(i)))
            self.add("load", ("load", t_addr, t_addr))
            self.add("valu", ("vbroadcast", v_c, t_addr))
            v_cached.append(v_c)
        v_diff_1_2 = self.alloc_scratch("v_diff_1_2", VLEN)
        self.add("valu", ("-", v_diff_1_2, v_cached[2], v_cached[1]))

        # 3. Load batch
        n_blocks_total = batch_size // VLEN
        v_indices = self.alloc_scratch("v_indices", batch_size)
        v_values = self.alloc_scratch("v_values", batch_size)
        t_addr_tmp = self.alloc_scratch(None)
        b_off_consts = [self.scratch_const(b * VLEN) for b in range(n_blocks_total)]
        for b in range(n_blocks_total):
            b_off_addr = b_off_consts[b]
            self.add("alu", ("+", t_addr_tmp, self.scratch["inp_indices_p"], b_off_addr))
            self.add("load", ("vload", v_indices + b * VLEN, t_addr_tmp))
            self.add("alu", ("+", t_addr_tmp, self.scratch["inp_values_p"], b_off_addr))
            self.add("load", ("vload", v_values + b * VLEN, t_addr_tmp))

        # 4. Rounds
        # Real limits from tests/frozen_problem.py
        OFFICIAL_LIMITS = {"alu": 12, "valu": 6, "load": 2, "store": 2, "flow": 1}
        N_INTERLEAVE = 21
        v_temps = []
        for i in range(N_INTERLEAVE):
            v_temps.append({
                "node": self.alloc_scratch(None, VLEN),
                "tmp1": self.alloc_scratch(None, VLEN),
                "tmp2": self.alloc_scratch(None, VLEN),
                "addr": self.alloc_scratch(None, VLEN),
            })

        for round in range(rounds):
            packer = DependencyAwarePacker(OFFICIAL_LIMITS)
            r_lev = round % (forest_height + 1)
            for b in range(n_blocks_total):
                t = v_temps[b % N_INTERLEAVE]
                cur_v_idx = v_indices + b * VLEN
                cur_v_val = v_values + b * VLEN
                r_idx = list(range(cur_v_idx, cur_v_idx + VLEN))
                r_val = list(range(cur_v_val, cur_v_val + VLEN))
                r_node = list(range(t["node"], t["node"] + VLEN))
                r_tmp1 = list(range(t["tmp1"], t["tmp1"] + VLEN))
                r_tmp2 = list(range(t["tmp2"], t["tmp2"] + VLEN))
                r_addr = list(range(t["addr"], t["addr"] + VLEN))

                # Node load
                if r_lev == 0:
                    packer.add("valu", ("^", cur_v_val, cur_v_val, v_cached[0]), r_val + list(range(v_cached[0], v_cached[0]+VLEN)), r_val)
                elif r_lev == 1:
                    packer.add("valu", ("%", t["tmp1"], cur_v_val, v_two), r_val + list(range(v_two, v_two+VLEN)), r_tmp1)
                    packer.add("valu", ("multiply_add", t["node"], t["tmp1"], v_diff_1_2, v_cached[1]), r_tmp1 + list(range(v_diff_1_2, v_diff_1_2+VLEN)) + list(range(v_cached[1], v_cached[1]+VLEN)), r_node)
                    packer.add("valu", ("^", cur_v_val, cur_v_val, t["node"]), r_val + r_node, r_val)
                elif r_lev == 2:
                    v_4 = get_v_const(4); v_6 = get_v_const(6)
                    packer.add("valu", ("==", t["tmp1"], cur_v_idx, v_4), r_idx + list(range(v_4, v_4+VLEN)), r_tmp1)
                    packer.add("flow", ("vselect", t["node"], t["tmp1"], v_cached[4], v_cached[3]), r_tmp1 + list(range(v_cached[4], v_cached[4]+VLEN)) + list(range(v_cached[3], v_cached[3]+VLEN)), r_node)
                    packer.add("valu", ("==", t["tmp1"], cur_v_idx, v_6), r_idx + list(range(v_6, v_6+VLEN)), r_tmp1)
                    packer.add("flow", ("vselect", t["tmp2"], t["tmp1"], v_cached[6], v_cached[5]), r_tmp1 + list(range(v_cached[6], v_cached[6]+VLEN)) + list(range(v_cached[5], v_cached[5]+VLEN)), r_tmp2)
                    packer.add("valu", ("<", t["tmp1"], v_4, cur_v_idx), list(range(v_4, v_4+VLEN)) + r_idx, r_tmp1)
                    packer.add("flow", ("vselect", t["node"], t["tmp1"], t["tmp2"], t["node"]), r_tmp1 + r_tmp2 + r_node, r_node)
                    packer.add("valu", ("^", cur_v_val, cur_v_val, t["node"]), r_val + r_node, r_val)
                else:
                    for vi in range(VLEN):
                        packer.add("alu", ("+", t["addr"] + vi, self.scratch["forest_values_p"], cur_v_idx + vi), [self.scratch["forest_values_p"], cur_v_idx + vi], [t["addr"] + vi])
                        packer.add("load", ("load", t["node"] + vi, t["addr"] + vi), [t["addr"] + vi], [t["node"] + vi])
                    packer.add("valu", ("^", cur_v_val, cur_v_val, t["node"]), r_val + r_node, r_val)

                # Hash
                for hi in range(6):
                    if hi in [0, 2, 4]:
                        v_m = v_mul[hi // 2]; v_a = v_add[hi // 2]
                        packer.add("valu", ("multiply_add", cur_v_val, cur_v_val, v_m, v_a), r_val + list(range(v_m, v_m+VLEN)) + list(range(v_a, v_a+VLEN)), r_val)
                    else:
                        op1, val1, op2, op3, val3 = HASH_STAGES[hi]
                        vh1 = v_hash_consts[(hi, 1)]; vh3 = v_hash_consts[(hi, 3)]
                        packer.add("valu", (op1, t["tmp1"], cur_v_val, vh1), r_val + list(range(vh1, vh1+VLEN)), r_tmp1)
                        packer.add("valu", (op3, t["tmp2"], cur_v_val, vh3), r_val + list(range(vh3, vh3+VLEN)), r_tmp2)
                        packer.add("valu", (op2, cur_v_val, t["tmp1"], t["tmp2"]), r_tmp1 + r_tmp2, r_val)

                # Traverse
                packer.add("valu", ("%", t["tmp1"], cur_v_val, v_two), r_val + list(range(v_two, v_two+VLEN)), r_tmp1)
                packer.add("valu", ("multiply_add", cur_v_idx, cur_v_idx, v_two, v_one), r_idx + list(range(v_two, v_two+VLEN)) + list(range(v_one, v_one+VLEN)), r_idx)
                packer.add("valu", ("+", cur_v_idx, cur_v_idx, t["tmp1"]), r_idx + r_tmp1, r_idx)
                packer.add("valu", ("<", t["tmp1"], cur_v_idx, v_n_nodes), r_idx + list(range(v_n_nodes, v_n_nodes+VLEN)), r_tmp1)
                packer.add("flow", ("vselect", cur_v_idx, t["tmp1"], cur_v_idx, v_zero), r_tmp1 + r_idx + list(range(v_zero, v_zero+VLEN)), r_idx)

                # Store back
                b_off_addr = b_off_consts[b]
                packer.add("alu", ("+", t["tmp1"], self.scratch["inp_indices_p"], b_off_addr), [self.scratch["inp_indices_p"], b_off_addr], [t["tmp1"]])
                packer.add("store", ("vstore", t["tmp1"], cur_v_idx), [t["tmp1"]] + r_idx, [])
                packer.add("alu", ("+", t["tmp1"], self.scratch["inp_values_p"], b_off_addr), [self.scratch["inp_values_p"], b_off_addr], [t["tmp1"]])
                packer.add("store", ("vstore", t["tmp1"], cur_v_val), [t["tmp1"]] + r_val, [])

            self.instrs.extend(packer.get_instrs())
            self.add("flow", ("pause",))

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


if __name__ == "__main__":
    unittest.main()
