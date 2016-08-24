------------------------------------------------------------------------
--[[ BiOneToManySequencer ]]--
-- Encapsulates a Module. 
-- Input is a sequence (a table) of tensors and an output length longer than the input.
-- Output is a sequence (a table) of tensors of the given length.
-- Applies the module to each element in the sequence multiple times until a STOP token is emmited.
-- Stop tokens do count as an output.
-- ?Handles both recurrent modules and non-recurrent modules.?
-- The sequences in a batch must have the same size and have the same output length
-- But the sequence length of each batch can vary.
------------------------------------------------------------------------
--assert(not nn.Sequencer, "update nnx package : luarocks install nnx")
local BiOneToManySequencer, parent = torch.class('nn.BiOneToManySequencer', 'nn.AbstractSequencer')
local _ = require 'moses'

BiOneToManySequencer.timeSize=3;

function BiOneToManySequencer:__init(module,nOut)
   parent.__init(self)
   if not torch.isTypeOf(module, 'nn.Module') then
      error"BiOneToManySequencer: expecting nn.Module instance at arg 1"
   end
   self.nOut=nOut
   
   -- we can decorate the module with a Recursor to make it AbstractRecurrent
   self.module = (not torch.isTypeOf(module, 'nn.AbstractRecurrent')) and nn.Recursor(module) or module
   self.moduleR = self.module:clone()
   -- backprop through time (BPTT) will be done online (in reverse order of forward)
   self.modules = {self.module, self.moduleR}
   
   self.output = {}
   self.tableoutput = {}
   self.tablegradInput = {}
   
   -- table of buffers used for evaluation
   self._output = {}
   -- so that these buffers aren't serialized :
   local _ = require 'moses'
   self.dpnn_mediumEmpty = _.clone(self.dpnn_mediumEmpty)
   table.insert(self.dpnn_mediumEmpty, '_output')
   -- default is to forget previous inputs before each forward()
   self._remember = 'neither'
end

function BiOneToManySequencer:isSTOP(state)
    assert(state:size(1)==1, "OneToMany currently only supports batches of size one")
    if (state[1][state:size(2)]>0.5) then
        return true
    else
        return false
    end
end

function BiOneToManySequencer:updateOutput(input)
   local nOut=self.nOut
   local nData
   if torch.isTensor(input) then
      nData = input:size(1)
      assert(input:size(2)==1, "OneToMany currently only supports batches of size one")
   else
      assert(torch.type(input) == 'table', "expecting input table")
      nData = #input
      assert(input[1]:size(1)==1, "OneToMany currently only supports batches of size one")
   end
   assert(nData<nOut,"number of outputs must be more than number of inputs")

   -- Note that the Sequencer hijacks the rho attribute of the rnn
   self.module:maxBPTTstep(nOut) --nStep??
   if self.train ~= false then 
      -- TRAINING
      self.inputRunMap={}
      self.inputRunMapR={}
      if not (self._remember == 'train' or self._remember == 'both') then
         self.module:forget()
      end
      
      self.tableoutput = {}
      self.tableoutputR = {}
      --TODO multithread this

      local dataStep=1
      local curData=input[dataStep]
      --print('updateOutput() start')
      local outStep=1
      while outStep<=nOut do
         local curOut
         while true do
            -- print ('updateOutput()',outStep, dataStep)
            local curTime=torch.Tensor(curData:size(1),BiOneToManySequencer.timeSize)
            curTime:select(2,1):fill( (nOut-outStep)/nOut )
            curTime:select(2,2):fill( (nData-dataStep)/nData )
            curTime:select(2,3):fill( nData/nOut )
            --print('data:',curData,'  time:',curTime)
            --print(torch.cat(curData,curTime,2))
            local curOut = self.module:updateOutput(torch.cat(curData,curTime,2))
            self.tableoutput[outStep]=curOut
            self.inputRunMap[outStep]=dataStep
            outStep = outStep+1
            if self:isSTOP(curOut) or outStep>nOut then
                 break
            end
         end
         if dataStep<nData then
             dataStep = dataStep+1
             curData = input[dataStep]
         else
             curData = input[nData] --we just use the last data value repeatedly if we run out
         end
      end

      ---------
      dataStep=nData
      curData=input[dataStep]
      outStep=nOut
      while outStep>=1 do
         local curOut
         while true do
            -- print ('updateOutput()',outStep, dataStep)
            local curTime=torch.Tensor(curData:size(1),BiOneToManySequencer.timeSize)
            curTime:select(2,1):fill( (nOut-outStep)/nOut )
            curTime:select(2,2):fill( (nData-dataStep)/nData )
            curTime:select(2,3):fill( nData/nOut )
            --print('data:',curData,'  time:',curTime)
            --print(torch.cat(curData,curTime,2))
            local curOut = self.moduleR:updateOutput(torch.cat(curData,curTime,2))
            self.tableoutputR[outStep]=curOut
            self.inputRunMapR[outStep]=dataStep
            outStep = outStep-1
            if self:isSTOP(curOut) or outStep==0 then
                 break
            end
         end
         if dataStep>1 then
             dataStep = dataStep-1
             curData = input[dataStep]
         else
             curData = input[1] --we just use the last data value repeatedly if we run out
         end
      end
      ----------
      for outStep=1,nOut do
          self.tableoutput[outStep]=torch.cat(self.tableoutput[outStep],self.tableoutputR[outStep],2)
      end


      if torch.isTensor(input) then
         self.output = torch.isTensor(self.output) and self.output or self.tableoutput[1].new()
         self.output:resize(nOut, unpack(self.tableoutput[1]:size():totable()))
         for step=1,nOut do
            self.output[step]=self.tableoutput[step]
         end
      else
            self.output= self.tableoutput
      end
   else 
      -- EVALUATION
      if not (self._remember == 'eval' or self._remember == 'both') then
         self.module:forget()
      end
      -- during evaluation, recurrent modules reuse memory (i.e. outputs)
      -- so we need to copy each output into our own table or tensor
      if torch.isTensor(input) then
          --TODO multithread this
          local dataStep=1
          local curData=input[dataStep]
          local outStep=1
          while outStep<=nOut do
             local curOut
             while true do
                --print ('updateOutput()-EVAL',outStep, dataStep)

                local curTime=torch.Tensor(curData:size(1),BiOneToManySequencer.timeSize)
                curTime:select(2,1):fill( (nOut-outStep)/nOut )
                curTime:select(2,2):fill( (nData-dataStep)/nData )
                curTime:select(2,3):fill( nData/nOut )
                curOut = self.module:updateOutput(torch.cat(curData,curTime,2))
                if outStep == 1 then
                   self.output = torch.isTensor(self.output) and self.output or output.new()
                   self.output:resize(nOut, unpack(output:size():totable()))
                end
                self.output[outStep]:narrow(2,1,curOut:size(2)):copy(curOut)     
                outStep = outStep+1
                if self:isSTOP(curOut) or outStep>nOut then
                     break
                end
             end
             if dataStep<nData then
                dataStep = dataStep+1
                 curData = input[dataStep]
             else
                 curData = input[nData] --we just use the last data value repeatedly if we run out
             end
          end
          ---------
          dataStep=nData
          curData=input[dataStep]
          outStep=nOut
          while outStep>=1 do
             local curOut
             while true do
                --print ('updateOutput()-EVAL',outStep, dataStep)

                local curTime=torch.Tensor(curData:size(1),BiOneToManySequencer.timeSize)
                curTime:select(2,1):fill( (nOut-outStep)/nOut )
                curTime:select(2,2):fill( (nData-dataStep)/nData )
                curTime:select(2,3):fill( nData/nOut )
                curOut = self.moduleR:updateOutput(torch.cat(curData,curTime,2))
                self.output[outStep]:narrow(2,curOut:size(2)+1,curOut:size(2)):copy(curOut)     
                outStep = outStep-1
                if self:isSTOP(curOut) or outStep==0 then
                     break
                end
             end
             if dataStep>1 then
                 dataStep = dataStep-1
                 curData = input[dataStep]
             else
                 curData = input[1] --we just use the last data value repeatedly if we run out
             end
          end
          ---------
          --for outStep=1,nOut do
          --    self.output[outStep]:cat(self.outputR[outStep],2)
          --end
      else
        assert(false,'unimplemented')
        --[[
         for step=1,nStep do
            self.tableoutput[step] = nn.rnn.recursiveCopy(
               self.tableoutput[step] or table.remove(self._output, 1), 
               self.module:updateOutput(input[step])
            )
         end --]]
          local dataStep=1
          local curData=input[dataStep]
          local outStep=1
          while outStep<=nOut do
             local curOut
             while true do
                --print ('updateOutput()-EVAL',outStep, dataStep)

                local curTime=torch.Tensor(curData:size(1),BiOneToManySequencer.timeSize)
                curTime:select(2,1):fill( (nOut-outStep)/nOut )
                curTime:select(2,2):fill( (nData-dataStep)/nData )
                curTime:select(2,3):fill( nData/nOut )
                curOut = self.module:updateOutput(torch.cat(curData,curTime,2))
                self.tableoutput[outStep] = nn.rnn.recursiveCopy(
                   self.tableoutput[outStep] or table.remove(self._output, 1), 
                   curOut
                )
                outStep = outStep+1
                if self:isSTOP(curOut) or outStep>nOut then
                     break
                end
             end
             if dataStep<nData then
                 dataStep = dataStep+1
                 curData = input[dataStep]
             else
                 curData = input[nData] --we just use the last data value repeatedly if we run out
             end
          end
         -- remove extra output tensors (save for later)
         for i=nOut+1,#self.tableoutput do
            table.insert(self._output, self.tableoutput[i])
            self.tableoutput[i] = nil
         end
         self.output = self.tableoutput
      end
   end
   
   --print('Bi out size:',self.output[1]:size(2))
   return self.output
end

function BiOneToManySequencer:updateGradInput(input, gradOutput)
   local nOut=self.nOut
   local nData
   if torch.isTensor(input) then
      assert(torch.isTensor(gradOutput), "expecting gradOutput Tensor since input is a Tensor")
      --assert(gradOutput:size(1) == input:size(1), "gradOutput should have as many elements as input")
      nData = input:size(1)
   else
      assert(torch.type(input) == 'table', "expecting gradOutput table")
      --assert(#gradOutput == #input, "gradOutput should have as many elements as input")
      nData = #input
   end
   assert(nData<nOut,"number of outputs must be more than number of inputs");
   assert(#gradOutput == nOut)
   -- back-propagate through time
   self.tablegradinput = {} --I have to save input ordering
   self.tablegradinputR = {} 
   for outStep=nOut,1,-1 do
      --print ('updateGradInput()',outStep, self.inputRunMap[outStep])
      local dataStep=self.inputRunMap[outStep]
        local curTime=torch.Tensor(input[dataStep]:size(1),BiOneToManySequencer.timeSize)
        curTime:select(2,1):fill( (nOut-outStep)/nOut )
        curTime:select(2,2):fill( (nData-dataStep)/nData )
        curTime:select(2,3):fill( nData/nOut )
      self.tablegradinput[outStep] = self.module:updateGradInput(torch.cat(input[dataStep],curTime,2), gradOutput[outStep]:narrow(2,1,gradOutput[outStep]:size(2)/2))
   end
   for outStep=1,nOut do
      --print ('updateGradInput()',outStep, self.inputRunMap[outStep])
      local dataStep=self.inputRunMapR[outStep]
        local curTime=torch.Tensor(input[dataStep]:size(1),BiOneToManySequencer.timeSize)
        curTime:select(2,1):fill( (nOut-outStep)/nOut )
        curTime:select(2,2):fill( (nData-dataStep)/nData )
        curTime:select(2,3):fill( nData/nOut )
      self.tablegradinputR[outStep] = self.moduleR:updateGradInput(torch.cat(input[dataStep],curTime,2), gradOutput[outStep]:narrow(2,gradOutput[outStep]:size(2)/2+1,gradOutput[outStep]:size(2)/2))
   end
   for outStep=1,nOut do
       self.tablegradinput[outStep] = torch.cat(self.tablegradinput[outStep],self.tablegradinputR[outStep],2)
   end
   
   if torch.isTensor(input) then
      self.gradInput = torch.isTensor(self.gradInput) and self.gradInput or self.tablegradinput[1].new()
      self.gradInput:resize(nOut, unpack(self.tablegradinput[1]:size():totable()))
      for step=1,nOut do
         self.gradInput[step]:copy(self.tablegradinput[step])
      end
   else
      self.gradInput = self.tablegradinput
   end

   return self.gradInput
end

function BiOneToManySequencer:accGradParameters(input, gradOutput, scale)
   local nOut=self.nOut
   local nData
   if torch.isTensor(input) then
      assert(torch.isTensor(gradOutput), "expecting gradOutput Tensor since input is a Tensor")
      --assert(gradOutput:size(1) == input:size(1), "gradOutput should have as many elements as input")
      nData = input:size(1)
   else
      assert(torch.type(input) == 'table', "expecting gradOutput table")
      --assert(#gradOutput == #input, "gradOutput should have as many elements as input")
      nData = #input
   end
   assert(nData<nOut,"number of outputs must be more than number of inputs");
   assert(#gradOutput == nOut)

   
   -- back-propagate through time 
   for outStep=nOut,1,-1 do
      --print ('accGradParameters()',outStep, self.inputRunMap[outStep])
      local dataStep=self.inputRunMap[outStep]
        local curTime=torch.Tensor(input[dataStep]:size(1),BiOneToManySequencer.timeSize)
        curTime:select(2,1):fill( (nOut-outStep)/nOut )
        curTime:select(2,2):fill( (nData-dataStep)/nData )
        curTime:select(2,3):fill( nData/nOut )
      self.module:accGradParameters(torch.cat(input[dataStep],curTime,2), gradOutput[outStep]:narrow(2,1,gradOutput[outStep]:size(2)/2), scale)
   end   
   for outStep=1,nOut do
      --print ('accGradParameters()',outStep, self.inputRunMap[outStep])
      local dataStep=self.inputRunMapR[outStep]
        local curTime=torch.Tensor(input[dataStep]:size(1),BiOneToManySequencer.timeSize)
        curTime:select(2,1):fill( (nOut-outStep)/nOut )
        curTime:select(2,2):fill( (nData-dataStep)/nData )
        curTime:select(2,3):fill( nData/nOut )
      self.moduleR:accGradParameters(torch.cat(input[dataStep],curTime,2), gradOutput[outStep]:narrow(2,gradOutput[outStep]:size(2)/2+1,gradOutput[outStep]:size(2)/2), scale)
   end   
end

function BiOneToManySequencer:accUpdateGradParameters(inputTable, gradOutputTable, lr)
   error"Not Implemented"  
end

-- Toggle to feed long sequences using multiple forwards.
-- 'eval' only affects evaluation (recommended for RNNs)
-- 'train' only affects training
-- 'neither' affects neither training nor evaluation
-- 'both' affects both training and evaluation (recommended for LSTMs)
-- Essentially, forget() isn't called on rnn module when remember is on
function BiOneToManySequencer:remember(remember)
   self._remember = (remember == nil) and 'both' or remember
   local _ = require 'moses'
   assert(_.contains({'both','eval','train','neither'}, self._remember), 
      "BiOneToManySequencer : unrecognized value for remember : "..self._remember)
   return self
end

function BiOneToManySequencer:training()
   if self.train == false then
      -- forget at the start of each training
      self:forget()
      -- empty temporary output table
      self._output = {}
      -- empty output table (tensor mem was managed by seq)
      self.tableoutput = nil
      self.tableoutputR = nil
   end
   parent.training(self)
end

function BiOneToManySequencer:evaluate()
   if self.train ~= false then
      -- forget at the start of each evaluation
      self:forget()
      -- empty output table (tensor mem was managed by rnn)
      self.tableoutput = {}
      self.tableoutputR = {}
   end
   parent.evaluate(self)
   assert(self.train == false)
end

function BiOneToManySequencer:clearState()
   if torch.isTensor(self.output) then
      self.output:set()
      self.gradInput:set()
   else
      self.output = {}
      self.gradInput = {}
   end
   self._output = {}
   self.tableoutput = {}
   self.tablegradinput = {}
   self.module:clearState()
   self.tableoutputR = {}
   self.tablegradinputR = {}
   self.moduleR:clearState()
end

BiOneToManySequencer.__tostring__ = nn.Decorator.__tostring__
