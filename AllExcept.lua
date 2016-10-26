--This applies the given module to all elements of the tensor execpt those specified.
--The 'except' values are appended to the end of the vector

local AllExcept, parent = torch.class('nn.AllExcept', 'nn.Module')

function AllExcept:__init(module,except)
   parent.__init(self)
   self.module=module
   self.except=torch.LongTensor(except)
   self.nIn=0
end

function AllExcept:setTheRest(size)
    local theRest = {}
    for i=1,size do
        local keep=true
        for j=1,self.except:size(1) do
            if (i==self.except[j]) then
                keep=false
                break
            end
        end
        if keep then
            table.insert(theRest,i)
        end
    end
    self.nIn=size
    self.theRest=torch.LongTensor(theRest)
end

function AllExcept:updateOutput(input)
    assert(input:nDimension()==2,'AllExcept expects 2 dimensional tensors')
    if self.nIn ~= input:size(2) then
        self:setTheRest(input:size(2))
    end
    local compute = input:index(2,self.theRest) 
    local pass = input:index(2,self.except)
    self.output = torch.cat(self.module:updateOutput(compute),pass,2)

   return self.output
end

function AllExcept:updateGradInput(input, gradOutput)
    assert(input:nDimension()==2,'AllExcept expects 2 dimensional tensors')
    if self.nIn ~= input:size(2) then
        self:setTheRest(input:size(2))
    end
    --print( self.theRest, self.except )
    local compute = input:index(2,self.theRest) 
    local pass = input:index(2,self.except)
    self.gradInput = torch.cat(self.module:updateGradInput(compute,gradOutput:narrow(2,1,gradOutput:size(2)-self.except:size(1))),gradOutput:narrow(2,gradOutput:size(2)-self.except:size(1),self.except:size(1)),2)

   return self.gradInput
end

function AllExcept:accGradParameters(input, gradOutput, scale)
    assert(input:nDimension()==2,'AllExcept expects 2 dimensional tensors')
    if self.nIn ~= input:size(2) then
        self:setTheRest(input:size(2))
    end
    local compute = input:index(2,self.theRest) 
    self.module:accGradParameters(compute,gradOutput:narrow(2,1,gradOutput:size(2)-self.except:size(1)),scale)

end


function AllExcept:clearState()
   return self.module.clearState()
end

function AllExcept:__tostring__()
  return 'AllExcept:' .. self.module:__tostring__()
end
