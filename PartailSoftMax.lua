--This applies a softmax to all but the last n elements of the tensor. It applies a sigmoid to the last n.

local PartailSoftMax, parent = torch.class('nn.PartailSoftMax', 'nn.Module')

function PartailSoftMax:__init(numToLeaveOff)
    parent.__init(self)
    self.numToLeaveOff=numToLeaveOff
end

function PartailSoftMax:updateOutput(input)
   --[[input.THNN.SoftMax_updateOutput(
      input:narrow(1,1,input:size(1)-1):cdata(),
      self.output:narrow(1,1,input:size(1)-1):cdata()
   )]]
   --print('size: ',input:size())
   --print ('in')
   --print (input)
   self.output = input:clone()
   local outNarrow = self.output:narrow(2,1,input:size(2)-self.numToLeaveOff)
   local inputMax,_ = torch.max(input:narrow(2,1,input:size(2)-self.numToLeaveOff),2)
   inputMax:mul(-1)
   outNarrow:add(  inputMax:expandAs(outNarrow) )
   outNarrow:exp()
   local z = torch.sum(outNarrow,2)
   --outNarrow:mm(mult:transpose(),outNarrow)
   outNarrow:cdiv(z:expandAs(outNarrow))

   --Do sigmoid for last elements
   local outNarrowE=self.output:narrow(2,input:size(2)-self.numToLeaveOff+1,self.numToLeaveOff)
   outNarrowE:mul(-1):exp():add(1):pow(-1)
   --print ('out')
   --print (self.output)
   return self.output
end

function PartailSoftMax:updateGradInput(input, gradOutput)
   --[[input.THNN.SoftMax_updateGradInput(
      input:narrow(1,1,input:size(1)-1):cdata(),
      gradOutput:narrow(1,1,input:size(1)-1):cdata(),
      self.gradInput:narrow(1,1,input:size(1)-1):cdata(),
      self.output:narrow(1,1,input:size(1)-1):cdata()
   )]]
   --print ('[given] gradOut')
   --print (gradOutput)
   --local gradOutNarrow = gradOutput:narrow(2,1,input:size(2)-self.numToLeaveOff)
   local outNarrow = self.output:narrow(2,1,input:size(2)-self.numToLeaveOff)
   self.gradInput = gradOutput:clone()
   local gradInNarrow = self.gradInput:narrow(2,1,input:size(2)-self.numToLeaveOff)
   
   local sum = torch.sum(torch.cmul(gradInNarrow, outNarrow),2)
   
   gradInNarrow:add( sum:mul(-1):expandAs(gradInNarrow) )
   gradInNarrow:cmul( outNarrow )

   --sigmoid for last elements
   local gradInNarrowE=self.gradInput:narrow(2,input:size(2)-self.numToLeaveOff+1,self.numToLeaveOff)
   local outputNarrowE=self.output:narrow(2,input:size(2)-self.numToLeaveOff+1,self.numToLeaveOff)
   local z = torch.mul(outputNarrowE,-1):add(1):cmul(outputNarrowE)
   gradInNarrowE:cmul(z);
   --print ('[output] gradIn')
   --print (self.gradInput)
   return self.gradInput
end
