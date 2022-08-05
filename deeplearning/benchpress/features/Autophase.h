// A modified version of the LLVM InstCount analysis pass which produces the
// features used in the work:
//
//   Huang, Q., Haj-Ali, A., Moses, W., Xiang, J., Stoica, I., Asanovic, K., &
//   Wawrzynek, J. (2019). Autophase: Compiler phase-ordering for hls with deep
//   reinforcement learning. FCCM. https://doi.org/10.1109/FCCM.2019.00049

#pragma once

#include <vector>
#include <string>

#include "llvm/Analysis/Passes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/Pass.h"

// NOTE(cummins): I would prefer not to pull in the LLVM namespace here but this
// is required for the Instruction.def macros.
using namespace llvm;

namespace autophase {

constexpr size_t kAutophaseFeatureDimensionality = 56;

class AutophaseInstCount : public FunctionPass, public InstVisitor<AutophaseInstCount> {
 public:
  // Get the counter values as a vector of integers.
  static void getFeatureVector(AutophaseInstCount* pass);
  static char ID;

  AutophaseInstCount() : FunctionPass(ID) {}
  friend class InstVisitor<AutophaseInstCount>;

 private:
  bool runOnFunction(Function& F) override;

  // Declare a counter variable and a getter function.
#define COUNTER(name, opname, unused_description)     \
  std::string get_op##name() const { return  opname;} \
  int64_t name = 0;                                   \
  int64_t get_##name() const { return name; }

  // Custom autophase counters.
  COUNTER(TotalInsts, "TotalInsts", "Number of instructions (of all types)");
  COUNTER(TotalBlocks, "TotalBlocks", "Number of basic blocks");
  COUNTER(BlockLow, "BlockLow", "Number of BB's with less than 15 instructions");
  COUNTER(BlockMid, "BlockMid", "Number of BB's with instructions between [15, 500]");
  COUNTER(BlockHigh, "BlockHigh", "Number of BB's with more than 500 instructions");
  COUNTER(TotalFuncs, "TotalFuncs", "Number of non-external functions");
  COUNTER(TotalMemInst, "TotalMemInst", "Number of memory instructions");
  COUNTER(BeginPhi, "BeginPhi", "# of Phi-nodes at beginning of BB");
  COUNTER(ArgsPhi, "ArgsPhi", "Total arguments to Phi nodes");
  COUNTER(BBNoPhi, "BBNoPhi", "# of BB's with no Phi nodes");
  COUNTER(BB03Phi, "BB03Phi", "# of BB's with Phi node # in range (0, 3]");
  COUNTER(BBHiPhi, "BBHiPhi", "# of BB's with more than 3 Phi nodes");
  COUNTER(BBNumArgsHi, "BBNumArgsHi", "# of BB where total args for phi nodes > 5");
  COUNTER(BBNumArgsLo, "BBNumArgsLo", "# of BB where total args for phi nodes is [1, 5]");
  COUNTER(testUnary, "testUnary", "Unary");
  COUNTER(binaryConstArg, "binaryConstArg", "Binary operations with a constant operand");
  COUNTER(callLargeNumArgs, "callLargeNumArgs", "# of calls with number of arguments > 4");
  COUNTER(returnInt, "returnInt", "# of calls that return an int");
  COUNTER(oneSuccessor, "oneSuccessor", "# of BB's with 1 successor");
  COUNTER(twoSuccessor, "twoSuccessor", "# of BB's with 2 successors");
  COUNTER(moreSuccessors, "moreSuccessors", "# of BB's with >2 successors");
  COUNTER(onePred, "onePred", "# of BB's with 1 predecessor");
  COUNTER(twoPred, "twoPred", "# of BB's with 2 predecessors");
  COUNTER(morePreds, "morePreds", "# of BB's with >2 predecessors");
  COUNTER(onePredOneSuc, "onePredOneSuc", "# of BB's with 1 predecessor and 1 successor");
  COUNTER(onePredTwoSuc, "onePredTwoSuc", "# of BB's with 1 predecessor and 2 successors");
  COUNTER(twoPredOneSuc, "twoPredOneSuc", "# of BB's with 2 predecessors and 1 successor");
  COUNTER(twoEach, "twoEach", "# of BB's with 2 predecessors and successors");
  COUNTER(moreEach, "moreEach", "# of BB's with >2 predecessors and successors");
  COUNTER(NumEdges, "NumEdges", "# of edges");
  COUNTER(CriticalCount, "CriticalCount", "# of critical edges");
  COUNTER(BranchCount, "BranchCount", "# of branches");
  COUNTER(numConstOnes, "numConstOnes", "# of occurrences of constant 1");
  COUNTER(numConstZeroes, "numConstZeroes", "# of occurrences of constant 0");
  COUNTER(const32Bit, "const32Bit", "# of occurrences of 32-bit integer constants");
  COUNTER(const64Bit, "const64Bit", "# of occurrences of 64-bit integer constants");
  COUNTER(UncondBranches, "UncondBranches", "# of unconditional branches");

// Generate opcode counters.
#define HANDLE_INST(N, OPCODE, CLASS) COUNTER(Num##OPCODE##Inst, #OPCODE, "Number of " #OPCODE " insts");

#include "llvm/IR/Instruction.def"


  void visitFunction(Function& F);

  void visitBasicBlock(BasicBlock& BB);

// Generate instruction visitors.
#define HANDLE_INST(N, OPCODE, CLASS) void visit##OPCODE(CLASS&);

#include "llvm/IR/Instruction.def"

  void visitInstruction(Instruction& I);
};

}  // namespace autophase
