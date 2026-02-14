
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
        self.vec_const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        if not vliw:
            instrs = []
            for engine, slot in slots:
                if engine is None: continue
                instrs.append({engine: [slot]})
            return instrs

        # Simple VLIW packing
        packed_instrs = []
        current_instr = defaultdict(list)

        for engine, slot in slots:
            if engine is None:
                if current_instr:
                    packed_instrs.append(dict(current_instr))
                    current_instr = defaultdict(list)
                continue

            if len(current_instr[engine]) < SLOT_LIMITS.get(engine, 64):
                current_instr[engine].append(slot)
            else:
                packed_instrs.append(dict(current_instr))
                current_instr = defaultdict(list)
                current_instr[engine].append(slot)

        if current_instr:
            packed_instrs.append(dict(current_instr))
        return packed_instrs

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
            addr = self.alloc_scratch(name or f"const_{val}")
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def scratch_const_vec(self, val, name=None):
        if val not in self.vec_const_map:
            s_addr = self.scratch_const(val)
            v_addr = self.alloc_scratch(name or f"v_const_{val}", VLEN)
            self.add("valu", ("vbroadcast", v_addr, s_addr))
            self.vec_const_map[val] = v_addr
        return self.vec_const_map[val]

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
        """
        Optimized vectorized implementation.
        """
        # Scratch space addresses for parameters
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)

        tmp_load = self.alloc_scratch("tmp_load")
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp_load, i))
            self.add("load", ("load", self.scratch[v], tmp_load))

        # Vectorized scratch space for idx and val
        assert batch_size % VLEN == 0
        n_blocks = batch_size // VLEN
        v_indices = self.alloc_scratch("v_indices", batch_size)
        v_values = self.alloc_scratch("v_values", batch_size)

        # Pre-load all indices and values into scratch
        for b in range(n_blocks):
            b_offset = b * VLEN
            # Load indices
            tmp_addr = self.alloc_scratch(f"tmp_addr_idx_{b}")
            self.add("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], self.scratch_const(b_offset)))
            self.add("load", ("vload", v_indices + b_offset, tmp_addr))
            # Load values
            tmp_addr_v = self.alloc_scratch(f"tmp_addr_val_{b}")
            self.add("alu", ("+", tmp_addr_v, self.scratch["inp_values_p"], self.scratch_const(b_offset)))
            self.add("load", ("vload", v_values + b_offset, tmp_addr_v))

        # Vector constants
        v_zero = self.scratch_const_vec(0)
        v_one = self.scratch_const_vec(1)
        v_two = self.scratch_const_vec(2)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
        self.add("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))

        P = 4 # Pipeline depth
        v_tmp1 = [self.alloc_scratch(f"v_tmp1_{p}", VLEN) for p in range(P)]
        v_tmp2 = [self.alloc_scratch(f"v_tmp2_{p}", VLEN) for p in range(P)]
        v_tmp3 = [self.alloc_scratch(f"v_tmp3_{p}", VLEN) for p in range(P)]
        v_node_val = [self.alloc_scratch(f"v_node_val_{p}", VLEN) for p in range(P)]
        tmp_node_addr = [[self.alloc_scratch(f"tmp_node_addr_{p}_{i}") for i in range(VLEN)] for p in range(P)]

        self.add("flow", ("pause",))

        for round in range(rounds):
            blocks_done = 0
            while blocks_done < n_blocks:
                group_size = min(4, n_blocks - blocks_done)
                cur_blocks = range(blocks_done, blocks_done + group_size)
                blocks_done += group_size

                body = []
                # Alu
                for b in cur_blocks:
                    p = b % P
                    for i in range(VLEN):
                        body.append(("alu", ("+", tmp_node_addr[p][i], self.scratch["forest_values_p"], v_indices + b*VLEN + i)))
                body.append((None, None))

                # Load
                for b in cur_blocks:
                    p = b % P
                    for i in range(VLEN):
                        body.append(("load", ("load", v_node_val[p] + i, tmp_node_addr[p][i])))
                body.append((None, None))

                # XOR
                for b in cur_blocks:
                    p = b % P
                    cur_v_val = v_values + b * VLEN
                    body.append(("valu", ("^", cur_v_val, cur_v_val, v_node_val[p])))
                body.append((None, None))

                # Hash
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    v_val1 = self.scratch_const_vec(val1)
                    v_val3 = self.scratch_const_vec(HASH_STAGES[hi][4])
                    for b in cur_blocks:
                        p = b % P
                        cur_v_val = v_values + b * VLEN
                        body.append(("valu", (op1, v_tmp1[p], cur_v_val, v_val1)))
                        body.append(("valu", (op3, v_tmp2[p], cur_v_val, v_val3)))
                    body.append((None, None))
                    for b in cur_blocks:
                        p = b % P
                        cur_v_val = v_values + b * VLEN
                        body.append(("valu", (op2, cur_v_val, v_tmp1[p], v_tmp2[p])))
                    body.append((None, None))

                # Update
                for b in cur_blocks:
                    p = b % P
                    cur_v_val = v_values + b * VLEN
                    body.append(("valu", ("%", v_tmp1[p], cur_v_val, v_two)))
                body.append((None, None))
                for b in cur_blocks:
                    p = b % P
                    body.append(("valu", ("==", v_tmp1[p], v_tmp1[p], v_zero)))
                body.append((None, None))
                for b in cur_blocks:
                    p = b % P
                    cur_v_idx = v_indices + b * VLEN
                    body.append(("flow", ("vselect", v_tmp3[p], v_tmp1[p], v_one, v_two)))
                body.append((None, None))
                for b in cur_blocks:
                    p = b % P
                    cur_v_idx = v_indices + b * VLEN
                    body.append(("valu", ("*", cur_v_idx, cur_v_idx, v_two)))
                body.append((None, None))
                for b in cur_blocks:
                    p = b % P
                    cur_v_idx = v_indices + b * VLEN
                    body.append(("valu", ("+", cur_v_idx, cur_v_idx, v_tmp3[p])))
                body.append((None, None))
                for b in cur_blocks:
                    p = b % P
                    cur_v_idx = v_indices + b * VLEN
                    body.append(("valu", ("<", v_tmp1[p], cur_v_idx, v_n_nodes)))
                body.append((None, None))
                for b in cur_blocks:
                    p = b % P
                    cur_v_idx = v_indices + b * VLEN
                    body.append(("flow", ("vselect", cur_v_idx, v_tmp1[p], cur_v_idx, v_zero)))
                body.append((None, None))

                self.instrs.extend(self.build(body, vliw=True))
            # Required to match with the yield in reference_kernel2
            self.instrs.append({"flow": [("pause",)]})

        # Store results back to memory
        for b in range(n_blocks):
            b_offset = b * VLEN
            tmp_addr = self.alloc_scratch(f"tmp_addr_store_idx_{b}")
            self.add("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], self.scratch_const(b_offset)))
            self.add("store", ("vstore", tmp_addr, v_indices + b * VLEN))

            tmp_addr_v = self.alloc_scratch(f"tmp_addr_store_val_{b}")
            self.add("alu", ("+", tmp_addr_v, self.scratch["inp_values_p"], self.scratch_const(b_offset)))
            self.add("store", ("vstore", tmp_addr_v, v_values + b * VLEN))

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
    # print(kb.instrs)

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
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
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
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
