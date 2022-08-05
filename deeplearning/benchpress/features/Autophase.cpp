//===-- AutophaseInstCount.cpp - Collects the count of all instructions ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass collects the count of all instructions and reports them
//
//===----------------------------------------------------------------------===//

#include "Autophase.h"

#include "llvm/Analysis/CFG.h"
#include "llvm/Support/raw_ostream.h"

namespace autophase {

char AutophaseInstCount::ID = 0;

bool AutophaseInstCount::runOnFunction(Function& F) {
  unsigned StartMemInsts = NumGetElementPtrInst + NumLoadInst + NumStoreInst + NumCallInst +
                           NumInvokeInst + NumAllocaInst;
  visit(F);
  unsigned EndMemInsts = NumGetElementPtrInst + NumLoadInst + NumStoreInst + NumCallInst +
                         NumInvokeInst + NumAllocaInst;
  TotalMemInst += EndMemInsts - StartMemInsts;
  getFeatureVector(this);
  return false;
}

void AutophaseInstCount::visitFunction(Function& F) {
  if (!F.isDeclaration()){
    ++TotalFuncs; 
  }
}

void AutophaseInstCount::visitBasicBlock(BasicBlock& BB) {
  ++TotalBlocks;
  Instruction* term = BB.getTerminator();
  unsigned numSuccessors = term->getNumSuccessors();
  for (int i = 0; i < numSuccessors; i++) {
    NumEdges++;
    if (isCriticalEdge(term, i)) {
      CriticalCount++;
    }
  }
  unsigned numPreds = 0;
  for (pred_iterator pi = pred_begin(&BB), E = pred_end(&BB); pi != E; ++pi) {
    numPreds++;
  }
  if (numSuccessors == 1) {
    oneSuccessor++;
  } else if (numSuccessors == 2) {
    twoSuccessor++;

  } else if (numSuccessors > 2) {
    moreSuccessors++;
  }
  if (numPreds == 1) {
    onePred++;
  } else if (numPreds == 2) {
    twoPred++;
  } else if (numPreds > 2) {
    morePreds++;
  }

  if (numPreds == 1 && numSuccessors == 1) {
    onePredOneSuc++;
  } else if (numPreds == 2 && numSuccessors == 1) {
    twoPredOneSuc++;
  } else if (numPreds == 1 && numSuccessors == 2) {
    onePredTwoSuc++;
  } else if (numPreds == 2 && numSuccessors == 2) {
    twoEach++;
  } else if (numPreds > 2 && numSuccessors > 2) {
    moreEach++;
  }

  unsigned tempCount = 0;
  bool isFirst = true;
  unsigned phiCount = 0;
  unsigned BBArgs = 0;
  for (Instruction& I : BB) {
    if (auto* bi = dyn_cast<BranchInst>(&I)) {
      BranchCount++;
      if (bi->isUnconditional()) {
        UncondBranches++;
      }
    }
    for (int i = 0; i < I.getNumOperands(); i++) {
      Value* v = I.getOperand(i);
      // Type* t = v->getType();
      if (auto* c = dyn_cast<Constant>(v)) {
        if (auto* ci = dyn_cast<ConstantInt>(c)) {
          APInt val = ci->getValue();
          unsigned bitWidth = val.getBitWidth();
          if (bitWidth == 32) {
            const32Bit++;
          } else if (bitWidth == 64) {
            const64Bit++;
          }
          if (val == 1) {
            numConstOnes++;
          } else if (val == 0) {
            numConstZeroes++;
          }
        }
      }
    }
    if (isa<CallInst>(I)) {
      if (cast<CallInst>(I).getNumArgOperands() > 4) {
        callLargeNumArgs++;
      }
      auto calledFunction = cast<CallInst>(I).getCalledFunction();
      if (calledFunction) {
        auto returnType = calledFunction->getReturnType();
        if (returnType) {
          if (returnType->isIntegerTy()) {
            returnInt++;
          }
        }
      }
    }
    if (isa<UnaryInstruction>(I)) {
      testUnary++;
    }
    if (isa<BinaryOperator>(I)) {
      if (isa<Constant>(I.getOperand(0)) || isa<Constant>(I.getOperand(1))) {
        binaryConstArg++;
      }
    }
    if (isFirst && isa<PHINode>(I)) {
      BeginPhi++;
    }
    if (isa<PHINode>(I)) {
      phiCount++;
      unsigned inc = cast<PHINode>(I).getNumIncomingValues();
      ArgsPhi += inc;
      BBArgs += inc;
    }
    isFirst = false;
    tempCount++;
  }
  if (phiCount == 0) {
    BBNoPhi++;
  } else if (phiCount <= 3) {
    BB03Phi++;
  } else {
    BBHiPhi++;
  }
  if (BBArgs > 5) {
    BBNumArgsHi++;
  } else if (BBArgs >= 1) {
    BBNumArgsLo++;
  }
  if (tempCount < 15) {
    BlockLow++;
  } else if (tempCount <= 500) {
    BlockMid++;
  } else {
    BlockHigh++;
  }
}

// Generate instruction visitors.
#define HANDLE_INST(N, OPCODE, CLASS)     \
  void AutophaseInstCount::visit##OPCODE(CLASS&) { \
    ++Num##OPCODE##Inst;                  \
    ++TotalInsts;                         \
  }

#include "llvm/IR/Instruction.def"

void AutophaseInstCount::visitInstruction(Instruction& I) {
  errs() << "Instruction Count does not know about " << I;
  llvm_unreachable(nullptr);
}

void AutophaseInstCount::getFeatureVector(AutophaseInstCount* pass) {

  outs() << pass->get_opBBNumArgsHi()          << " : " << pass->get_BBNumArgsHi()          << "\n";
  outs() << pass->get_opBBNumArgsLo()          << " : " << pass->get_BBNumArgsLo()          << "\n";
  outs() << pass->get_oponePred()              << " : " << pass->get_onePred()              << "\n";
  outs() << pass->get_oponePredOneSuc()        << " : " << pass->get_onePredOneSuc()        << "\n";
  outs() << pass->get_oponePredTwoSuc()        << " : " << pass->get_onePredTwoSuc()        << "\n";
  outs() << pass->get_oponeSuccessor()         << " : " << pass->get_oneSuccessor()         << "\n";
  outs() << pass->get_optwoPred()              << " : " << pass->get_twoPred()              << "\n";
  outs() << pass->get_optwoPredOneSuc()        << " : " << pass->get_twoPredOneSuc()        << "\n";
  outs() << pass->get_optwoEach()              << " : " << pass->get_twoEach()              << "\n";
  outs() << pass->get_optwoSuccessor()         << " : " << pass->get_twoSuccessor()         << "\n";
  outs() << pass->get_opmorePreds()            << " : " << pass->get_morePreds()            << "\n";
  outs() << pass->get_opBB03Phi()              << " : " << pass->get_BB03Phi()              << "\n";
  outs() << pass->get_opBBHiPhi()              << " : " << pass->get_BBHiPhi()              << "\n";
  outs() << pass->get_opBBNoPhi()              << " : " << pass->get_BBNoPhi()              << "\n";
  outs() << pass->get_opBeginPhi()             << " : " << pass->get_BeginPhi()             << "\n";
  outs() << pass->get_opBranchCount()          << " : " << pass->get_BranchCount()          << "\n";
  outs() << pass->get_opreturnInt()            << " : " << pass->get_returnInt()            << "\n";
  outs() << pass->get_opCriticalCount()        << " : " << pass->get_CriticalCount()        << "\n";
  outs() << pass->get_opNumEdges()             << " : " << pass->get_NumEdges()             << "\n";
  outs() << pass->get_opconst32Bit()           << " : " << pass->get_const32Bit()           << "\n";
  outs() << pass->get_opconst64Bit()           << " : " << pass->get_const64Bit()           << "\n";
  outs() << pass->get_opnumConstZeroes()       << " : " << pass->get_numConstZeroes()       << "\n";
  outs() << pass->get_opnumConstOnes()         << " : " << pass->get_numConstOnes()         << "\n";
  outs() << pass->get_opUncondBranches()       << " : " << pass->get_UncondBranches()       << "\n";
  outs() << pass->get_opbinaryConstArg()       << " : " << pass->get_binaryConstArg()       << "\n";
  outs() << pass->get_opNumAShrInst()          << " : " << pass->get_NumAShrInst()          << "\n";
  outs() << pass->get_opNumAddInst()           << " : " << pass->get_NumAddInst()           << "\n";
  outs() << pass->get_opNumAllocaInst()        << " : " << pass->get_NumAllocaInst()        << "\n";
  outs() << pass->get_opNumAndInst()           << " : " << pass->get_NumAndInst()           << "\n";
  outs() << pass->get_opBlockMid()             << " : " << pass->get_BlockMid()             << "\n";
  outs() << pass->get_opBlockLow()             << " : " << pass->get_BlockLow()             << "\n";
  outs() << pass->get_opNumBitCastInst()       << " : " << pass->get_NumBitCastInst()       << "\n";
  outs() << pass->get_opNumBrInst()            << " : " << pass->get_NumBrInst()            << "\n";
  outs() << pass->get_opNumCallInst()          << " : " << pass->get_NumCallInst()          << "\n";
  outs() << pass->get_opNumGetElementPtrInst() << " : " << pass->get_NumGetElementPtrInst() << "\n";
  outs() << pass->get_opNumICmpInst()          << " : " << pass->get_NumICmpInst()          << "\n";
  outs() << pass->get_opNumLShrInst()          << " : " << pass->get_NumLShrInst()          << "\n";
  outs() << pass->get_opNumLoadInst()          << " : " << pass->get_NumLoadInst()          << "\n";
  outs() << pass->get_opNumMulInst()           << " : " << pass->get_NumMulInst()           << "\n";
  outs() << pass->get_opNumOrInst()            << " : " << pass->get_NumOrInst()            << "\n";
  outs() << pass->get_opNumPHIInst()           << " : " << pass->get_NumPHIInst()           << "\n";
  outs() << pass->get_opNumRetInst()           << " : " << pass->get_NumRetInst()           << "\n";
  outs() << pass->get_opNumSExtInst()          << " : " << pass->get_NumSExtInst()          << "\n";
  outs() << pass->get_opNumSelectInst()        << " : " << pass->get_NumSelectInst()        << "\n";
  outs() << pass->get_opNumShlInst()           << " : " << pass->get_NumShlInst()           << "\n";
  outs() << pass->get_opNumStoreInst()         << " : " << pass->get_NumStoreInst()         << "\n";
  outs() << pass->get_opNumSubInst()           << " : " << pass->get_NumSubInst()           << "\n";
  outs() << pass->get_opNumTruncInst()         << " : " << pass->get_NumTruncInst()         << "\n";
  outs() << pass->get_opNumXorInst()           << " : " << pass->get_NumXorInst()           << "\n";
  outs() << pass->get_opNumZExtInst()          << " : " << pass->get_NumZExtInst()          << "\n";
  outs() << pass->get_opTotalBlocks()          << " : " << pass->get_TotalBlocks()          << "\n";
  outs() << pass->get_opTotalInsts()           << " : " << pass->get_TotalInsts()           << "\n";
  outs() << pass->get_opTotalMemInst()         << " : " << pass->get_TotalMemInst()         << "\n";
  outs() << pass->get_opTotalFuncs()           << " : " << pass->get_TotalFuncs()           << "\n";
  outs() << pass->get_opArgsPhi()              << " : " << pass->get_ArgsPhi()              << "\n";
  outs() << pass->get_optestUnary()            << " : " << pass->get_testUnary()            << "\n";
  return;
}

static RegisterPass<AutophaseInstCount> X("autophase", "Collect autophase paper features",
                             false /* Only looks at CFG */,
                             true /* Analysis Pass */);
}  // namespace autophase
