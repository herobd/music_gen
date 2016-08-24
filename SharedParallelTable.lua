local SharedParallelTable, parent = torch.class('nn.SharedParallelTable', 'nn.Module')


function SharedParallelTable:__init(unit, maxT)
    parent.__init(self)
    self.output    = {}
    self.gradInput = {}
    self.unit = unit
    self.maxT = maxT
    self.clones = {}
    self:reset()
end

function cloneManyTimes(net, T)
    --[[ Clone a network module T times, adapted from https://github.com/karpathy/char-rnn
    ARGS:
      - `net`    : network module to be cloned
      - `T`      : integer, number of clones
    RETURNS:
      - `clones` : table, list of clones
    ]]
    local clones = {}
    local params, gradParams = net:parameters()
    local mem = torch.MemoryFile("w"):binary()
    mem:writeObject(net)
    for t = 1, T do
        local reader = torch.MemoryFile(mem:storage(), "r"):binary()
        local clone = reader:readObject()
        reader:close()
        local cloneParams, cloneGradParams = clone:parameters()
        if params then
            for i = 1, #params do
                cloneParams[i]:set(params[i])
                cloneGradParams[i]:set(gradParams[i])
            end
        end
        clones[t] = clone
        collectgarbage()
    end
    mem:close()
    return clones
end

function SharedParallelTable:reset(stdv)
    local params, _ = self:parameters()
    if not params then return end
    for i = 1, #params do
        if i % 2 == 1 then
            params[i]:uniform(-0.08, 0.08)
        else
            params[i]:zero()
        end
    end
end


function SharedParallelTable:type(type)
    assert(#self.clones == 0, 'Function type() should not be called after cloning.')
    parent.type(self.unit, type)
    return self
end


function SharedParallelTable:parameters()
    return self.unit:parameters()
end


function SharedParallelTable:updateOutput(input)
    assert(type(input) == 'table')

    if #self.clones == 0 then
        self.clones = cloneManyTimes(self.unit, self.maxT)
    end

    self.output = {}
    local T = #input
    for t = 1, T do
        if self.train then
            self.output[t] = self.clones[t]:updateOutput(input[t]):clone()
        else
            self.output[t] = self.unit:updateOutput(input[t]):clone()
        end
    end
    return self.output
end


function SharedParallelTable:updateGradInput(input, gradOutput)
    assert(type(gradOutput) == 'table' and #input == #gradOutput)

    local T = #input
    for t = 1, T do
        self.gradInput[t] = self.clones[t]:updateGradInput(input[t], gradOutput[t])
    end
    return self.gradInput
end


function SharedParallelTable:accGradParameters(input, gradOutput, scale)
    local T = #input
    for t = 1, T do
        self.clones[t]:accGradParameters(input[t], gradOutput[t], scale)
    end
end
