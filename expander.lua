--TODO see if it will overfit. Simplified dataset (length=length). ReLU

require 'paths'
require 'rnn'
require 'OneToManySequencer'
require 'BiOneToManySequencer'
require 'PartailSoftMax'
require 'SharedParallelTable'
require 'AllExcept'
local utf8 = require 'lua-utf8'
local dl = require 'dataload'

version = 0

function isUpper(s)
    return s == utf8.upper(s)
end

function file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

function lines_from(file)
  if not file_exists(file) then return {} end
  local lines = {}
  for line in io.lines(file) do 
    lines[#lines + 1] = line
  end
  --print (lines)
  return lines
end

function shuffleTables( t1, t2 )
    local rand = math.random 
    assert( t1, "shuffleTables() expected a table, got nil" )
    assert( #t1==#t2, "shuffleTables() expected a tables of same size" )
    local iterations = #t1
    local j
    
    for i = iterations, 2, -1 do
        j = rand(i)
        t1[i], t1[j] = t1[j], t1[i]
        t2[i], t2[j] = t2[j], t2[i]
    end
end

function input_labels_from_file(file,splitTest)
    local lines = lines_from(file)
    local inputCounter=0
    local labelCounter=1
    local inputWords={}
    local labelWords={}
    local inputChars={} --{ [0]=' ' }
    local labelChars={}
    for i,line in pairs(lines) do
        local s = {}
        for x in utf8.gmatch(line, "%S+") do
            table.insert(s,x)
        end
        --print(s)
        local input=s[1]
        table.insert(inputWords,input)
        for c in utf8.gmatch(input,".") do
            if not inputChars[c] then
                inputChars[c]=inputCounter
                inputCounter = inputCounter+1
            end
        end
        local label=s[2]
        table.insert(labelWords,label)
        for uc in utf8.gmatch(label,".") do
            local c = utf8.lower(uc);
            if not labelChars[c] then 
                labelChars[c]=labelCounter
                labelCounter = labelCounter+1
            end
        end
    end
    
    --shuffleTables(inputWords,labelWords)
    local splitAt = math.floor(splitTest*#inputWords)
    
    trainInputVectors= {}
    trainInputWords= {}
    trainLabelTables= {}
    trainLabelTensors= {}
    trainLabelWords={}
    testInputVectors= {}
    testInputWords= {}
    testLabelTables= {}
    testLabelTensors= {}
    testLabelWords= {}
    
    
    
    for i = 1, splitAt-1 do
        local paddedLen = utf8.len(inputWords[i]) --*4 + 3 --math.max(utf8.len(inputWords[i]),utf8.len(labelWords[i]))+2 --math.max(utf8.len(inputWords[i])*3+5,10)
        table.insert(testInputWords,inputWords[i])
        local inputVec = torch.zeros(paddedLen,inputCounter)
        for t =1, utf8.len(inputWords[i]) do
            local c=utf8.sub(inputWords[i],t,t)
            --print (inputVec,t,inputWords[i],c,inputChars[c],inputCounter)
            inputVec[t][inputChars[c]+1]=1
            
        end
        --[[for t =1+utf8.len(inputWords[i]), paddedLen do
            inputVec[t][1]=1
        end]]
        table.insert(testInputVectors,inputVec)
        
        table.insert(testLabelWords,labelWords[i])
        local labelTable = {}
        local labelTensors = {}
        local lastChar='$'
        for t =1, utf8.len(labelWords[i]) do
            local labelTensor = torch.Tensor(1,labelCounter+1):zero()
            local uc=utf8.sub(labelWords[i],t,t)
            local c=utf8.lower(uc)
            labelTable[t]=labelChars[c]
            labelTensor[1][labelChars[c]]=1
            if isUpper(uc) then
                labelTensor[1][labelCounter]=1
            end
            if lastChar~=c then --this is a start char
                labelTensor[1][labelCounter+1]=1
            end
            lastChar=c
            labelTensors[t]=labelTensor
        end
        table.insert(testLabelTables,labelTable)
        table.insert(testLabelTensors,labelTensors)
    end
    for i = splitAt, #inputWords do
        local paddedLen = utf8.len(inputWords[i]) --*4 + 3 --math.max(utf8.len(inputWords[i]),utf8.len(labelWords[i]))+2 --math.max(utf8.len(inputWords[i])*3+5,10)
        table.insert(trainInputWords,inputWords[i])
        --print(i,inputWords[i])
        local inputVec = torch.zeros(paddedLen,inputCounter)
        for t =1, utf8.len(inputWords[i]) do
            local c=utf8.sub(inputWords[i],t,t)
            inputVec[t][inputChars[c]+1]=1
        end
        --[[for t =1+utf8.len(inputWords[i]), paddedLen do
            inputVec[t][1]=1
        end]]
        table.insert(trainInputVectors,inputVec)
        
        table.insert(trainLabelWords,labelWords[i])
        local labelTable = {}
        local labelTensors = {}
        local lastChar='$'
        for t =1, utf8.len(labelWords[i]) do
            local labelTensor = torch.Tensor(1,labelCounter+1):zero()
            local uc=utf8.sub(labelWords[i],t,t)
            local c=utf8.lower(uc)
            labelTable[t]=labelChars[c]
            labelTensor[1][labelChars[c]]=1
            if isUpper(uc) then
                labelTensor[1][labelCounter]=1
            end
            if lastChar~=c then --this is a startChar
                labelTensor[1][labelCounter+1]=1
            end
            lastChar=c
            labelTensors[t]=labelTensor
        end
        table.insert(trainLabelTables,labelTable)
        table.insert(trainLabelTensors,labelTensors)
    end
    
    return trainInputVectors, trainInputWords, trainLabelTensors, trainLabelWords,
           testInputVectors, testInputWords, testLabelTensors, testLabelWords,
           inputCounter, labelCounter-1, labelChars
end

--This both breaks the tables into batches, but formats them into what warp-ctc expects
function make_batches(inputVectors, inputWords, labelTables, labelWords, batchSize)
    local batchesRet={}
    --local labelRet={}
    
    local batchInput={}
    local batchLabel={}
    local batchInputW={}
    local batchLabelW={}
    local maxLength=0
    for i=1, #inputVectors do
    --for i=1, math.min(100,#inputVectors) do
        table.insert(batchInput,inputVectors[i])
        table.insert(batchLabel,labelTables[i])
        table.insert(batchInputW,inputWords[i])
        table.insert(batchLabelW,labelWords[i])
        if inputVectors[i]:size(1)>maxLength then
            maxLength=inputVectors[i]:size(1)
        end
        if i%batchSize==0 then --format batch
            --
            theBatch={inputs={}, labels=batchLabel, sizes={}, inputWords=batchInputW, labelWords=batchLabelW}
            for t=1, maxLength do
                theBatch.inputs[t]=torch.Tensor(batchSize,inputVectors[i]:size(2))
            end
            
            --this leaves some Tensors uninitailized, but they shouldn't be used anyways
            for n,vec in pairs(batchInput) do
                theBatch.sizes[n]=vec:size(1)
                
                for t=1, vec:size(1) do
                    theBatch.inputs[t][n]=vec[t]
                end
            end
            
            table.insert(batchesRet,theBatch)
            
            batchInput={}
            batchLabel={}
            batchInputW={}
            batchLabelW={}
            maxLength=0
        end
    end
    
    return batchesRet
end

function table_invert(t)
   local s={}
   for k,v in pairs(t) do
     s[v]=k
   end
   return s
end

function decode(activations,indexToChars,nInBatch)
    local decoded=''
    local blank=true
    local lastC
    for t,v in pairs(activations) do
        local acts = v[nInBatch]
        local cap = acts[#indexToChars +1]>0.5
        local str = acts[#indexToChars +2]>0.5
        acts[#indexToChars +1]=0;
        acts[#indexToChars +2]=0;
        local maxs, indices = torch.max(acts,1)
        local c = indexToChars[ indices[1] ]
        if cap then
            c=utf8.upper(c)
        end
        if str then
            c='^'..c
        end
        decoded = decoded..c
    end
    return decoded
end


--[[ command line arguments ]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a English to IPA translator')
cmd:text('Example:')
cmd:text('th simple_trans.lua --cuda --device 2 --progress --cutoff 4 ')
cmd:text("th simple_trans.lua --progress --cuda --lstm  --hiddensize '{200,200}' --batchsize 20 --startlr 1 --cutoff 5 --maxepoch 13 --schedule '{[5]=0.5,[6]=0.25,[7]=0.125,[8]=0.0625,[9]=0.03125,[10]=0.015625,[11]=0.0078125,[12]=0.00390625}'")
cmd:text("th simple_trans.lua --progress --cuda --lstm  --uniform 0.04 --hiddensize '{1500,1500}' --batchsize 20 --startlr 1 --cutoff 10 --maxepoch 50 --schedule '{[15]=0.87,[16]=0.76,[17]=0.66,[18]=0.54,[19]=0.43,[20]=0.32,[21]=0.21,[22]=0.10}' -dropout 0.65")
cmd:text('Options:')
-- training
cmd:option('--data', 'data.txt', 'path to dataset')
cmd:option('--startlr', 0.05, 'learning rate at t=0')
cmd:option('--minlr', 0.00001, 'minimum learning rate')
cmd:option('--saturate', 400, 'epoch at which linear decayed LR will reach minlr')
cmd:option('--schedule', '', 'learning rate schedule. e.g. {[5] = 0.004, [6] = 0.001}')
cmd:option('--momentum', 0.9, 'momentum')
cmd:option('--maxnormout', -1, 'max l2-norm of each layer\'s output neuron weights')
cmd:option('--cutoff', -1, 'max l2-norm of concatenation of all gradParam tensors')
--cmd:option('--batchSize', 32, 'number of examples per batch')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--device', 1, 'sets the device (GPU) to use')
cmd:option('--maxepoch', 1000, 'maximum number of epochs to run')
cmd:option('--earlystop', 50, 'maximum number of epochs to wait to find a better local minima for early-stopping')
cmd:option('--progress', false, 'print progress bar')
cmd:option('--silent', false, 'don\'t print anything to stdout')
cmd:option('--uniform', 0.1, 'initialize parameters using uniform distribution between -uniform and uniform. -1 means default initialization')
-- rnn layer 
cmd:option('--lstm', false, 'use Long Short Term Memory (nn.LSTM instead of nn.Recurrent)')
cmd:option('--gru', false, 'use Gated Recurrent Units (nn.GRU instead of nn.Recurrent)')
--cmd:option('--seqlen', 5, 'sequence length : back-propagate through time (BPTT) for this many time-steps')
cmd:option('--hiddensize', '{200}', 'number of hidden units used at output of each recurrent layer. When more than one is specified, RNN/LSTMs/GRUs are stacked')
cmd:option('--dropout', 0, 'apply dropout with this probability after each rnn layer. dropout <= 0 disables it.')
-- data
cmd:option('--batchsize', 1, 'number of examples per batch')
cmd:option('--splittest', 0.1, 'What portion of the dataset should be reserved for testing (validation).')
cmd:option('--trainsize', -1, 'number of train examples seen between each epoch')
cmd:option('--validsize', -1, 'number of valid examples used for early stopping and cross-validation') 
cmd:option('--savepath', paths.concat(dl.SAVE_PATH, 'simple_trans'), 'path to directory where experiment log (includes model) will be saved')
cmd:option('--id', '', 'id string of this experiment (used to name output file) (defaults to a unique id)')

cmd:text()
local opt = cmd:parse(arg or {})
opt.hiddensize = loadstring(" return "..opt.hiddensize)()
opt.schedule = loadstring(" return "..opt.schedule)()
if not opt.silent then
   table.print(opt)
end
opt.id = opt.id == '' and ('ptb' .. ':' .. dl.uniqueid()) or opt.id

--[[ data set ]]--
print('loading dataset')
trainInputVectors, trainInputWords, trainLabelTables, trainLabelWords, 
testInputVectors, testInputWords, testLabelTables, testLabelWords, 
inputsize, outputsize, labelChars = input_labels_from_file(opt.data,opt.splittest)
local indexToChars= table_invert(labelChars)
print(indexToChars)
print('making batches')
trainSet = make_batches(trainInputVectors,trainInputWords, trainLabelTables,trainLabelWords, opt.batchsize)
validSet = make_batches(testInputVectors,testInputWords, testLabelTables,testLabelWords, opt.batchsize)
print('done preparing dataset')
--print ('example batch')
--print(trainSet[1])

--[[local trainset, validset, testset = dl.loadPTB({opt.batchsize,1,1})
if not opt.silent then 
   print("Vocabulary size : "..#trainset.ivocab) 
   print("Train set split into "..opt.batchsize.." sequences of length "..trainset:size())
end--]]

--[[ rnn model ]]--

local lm = nn.Sequential()

-- input layer (i.e. word embedding space)
-- is this needed? not for hw atleast....
--local lookup = nn.LookupTable(#trainset.ivocab, opt.hiddensize[1])
--lookup.maxnormout = -1 -- prevent weird maxnormout behaviour
--lm:add(lookup) -- input is seqlen x batchsize
if opt.dropout > 0 and not opt.gru then  -- gru has a dropout option
   lm:add(nn.NaN(nn.Dropout(opt.dropout)))
end

--lm:add(nn.SplitTable(1)) -- tensor to table of tensors, I've preprocessed the data to this already

--input needs to be table (time) of tensors (batch x inputlayer)

-- rnn layers
local stepmodule = nn.Sequential() -- applied at each time-step
--local inputsize = trainInputVectors[0]:size(2) --the number of different chars
local lastLayerSize = inputsize
for i,hiddensize in ipairs(opt.hiddensize) do 
   lastLayerSize = lastLayerSize+nn.OneToManySequencer.timeSize
   local rnn
   if opt.gru then -- Gated Recurrent Units
      rnn = nn.GRU(lastLayerSize, hiddensize, nil, opt.dropout/2)
   elseif opt.lstm then -- Long Short Term Memory units
      require 'nngraph'
      nn.FastLSTM.usenngraph = true -- faster
      rnn = nn.FastLSTM(lastLayerSize, hiddensize)
   else -- simple recurrent neural network
      local rm =  nn.Sequential() -- input is {x[t], h[t-1]}
         :add(nn.ParallelTable()
            :add(i==1 and nn.Identity() or nn.Linear(lastLayerSize, hiddensize)) -- input layer
            :add(nn.Linear(hiddensize, hiddensize))) -- recurrent layer
         :add(nn.CAddTable()) -- merge
         :add(nn.Sigmoid()) -- transfer
      rnn = nn.Recurrence(rm, hiddensize, 1)
   end

   stepmodule:add(nn.NaN(rnn))
   
   if opt.dropout > 0 then
      stepmodule:add(nn.Dropout(opt.dropout))
   end
   
   lastLayerSize = hiddensize
end

-- output layer (not input size, that variable is holding the last layers output size)
stepmodule:add(nn.Linear(lastLayerSize, outputsize+1)) --+1 for END token
--stepmodule:add(nn.PartailSoftMax(1))

--if opt.cuda then
--    gpu_ctc(acts, grads, labels, sizes)
--else
--    cpu_ctc(acts, grads, labels, sizes)
   

-- encapsulate stepmodule into a Sequencer
local timeLen=utf8.len(trainLabelWords[1])
print("creating BiOneToMany with len:",timeLen)
lm:add(nn.NaN(nn.BiOneToManySequencer(stepmodule,timeLen)))

local endmodule = nn.Sequential() -- applied at each time-step, but not recurrently
local except = {} --we exclude the flags from the combination layer so the gradient flows back directly
except[1]=outputsize+1 --end flag
except[2]=(outputsize+1)*2 --start flag
endmodule:add(nn.AllExcept(nn.Linear(outputsize*2, outputsize),except))
endmodule:add(nn.PartailSoftMax(2)) --we exclude the flags from the SoftMax
lm:add(nn.SharedParallelTable(endmodule,timeLen))

-- remember previous state between batches: Not needed as each batch is a line
--lm:remember((opt.lstm or opt.gru) and 'both' or 'eval')

if not opt.silent then
   print"Test Model:"
   print(lm)
end

if opt.uniform > 0 then
   for k,param in ipairs(lm:parameters()) do
      param:uniform(-opt.uniform, opt.uniform)
   end
end

--[[ loss function ]]--

--local crit = nn.ClassNLLCriterion()
local crit = nn.BCECriterion()

-- we have already set up the labels how they need to be
--local targetmodule = nn.SplitTable(1)
if opt.cuda then
   --targetmodule = nn.Sequential()
   --   :add(nn.Convert())
   --   :add(targetmodule)
end

local outputConverter = nn.SplitTable(1)
if opt.cuda then
   outputConverter = nn.Sequential()
      :add(nn.Convert())
      :add(outputConverter)
end
 
local criterion = nn.SequencerCriterion(crit)

--[[ CUDA ]]--

if opt.cuda then
   require 'cunn'
   cutorch.setDevice(opt.device)
   lm:cuda()
   criterion:cuda()
   --targetmodule:cuda()
   outputConverter:cuda()
end

--[[ experiment log ]]--

-- is saved to file every time a new validation minima is found
local xplog = {}
xplog.opt = opt -- save all hyper-parameters and such
xplog.dataset = 'cmu-ipa'
--xplog.vocab = trainset.vocab
-- will only serialize params
xplog.model = nn.Serial(lm)
xplog.model:mediumSerial()
--xplog.model = lm
xplog.criterion = criterion
xplog.outputConverter = outputConverter
-- keep a log of NLL for each epoch
xplog.trainppl = {}
xplog.valppl = {}
-- will be used for early-stopping
xplog.minvalppl = 99999999
xplog.epoch = 0
local ntrial = 0
paths.mkdir(opt.savepath)

local epoch = 1
opt.lr = opt.startlr
local opt_trainsize = #trainSet --opt_trainsize == -1 and #trainSet or opt_trainsize
local opt_validsize = #validSet --opt_validsize == -1 and #validSet or opt_validsize
while opt.maxepoch <= 0 or epoch <= opt.maxepoch do
   print("")
   print("Epoch #"..epoch.." :")

   -- 1. training
   
   local a = torch.Timer()
   lm:training()
   local sumErr = 0
   for i, batch in pairs(trainSet) do
      --targets = targetmodule:forward(targets) --?
      
      -- forward
      local outputs = lm:forward(batch.inputs) --inputs is time
      --[[local outputs = {}
      --strip off the STOP feature for this particular dataset
      for i,o in pairs(outputsWithSTOP)  do
          outputs[i] = o:narrow(2,1,o:size(2)-1)
      end
      --]]
      local targets = batch.labels[1]
      --print ('to:',targets,outputs)
      
      assert(#targets == #outputs,"ERROR, targets and outputs different lengths (time)")
      local err = criterion:forward(outputs, targets)
      sumErr = sumErr+err
      
      --print("acts")
      --sizes comes from earlier
      local gradOutputs = criterion:backward(outputs, targets)
      --print('gradout:',gradOutputs)
      lm:zeroGradParameters()
      lm:backward(batch.inputs, gradOutputs)
      --print("backwrds done")
      -- update
      lm:updateGradParameters(opt.momentum) -- affects gradParams
      lm:updateParameters(opt.lr) -- affects params
      lm:maxParamNorm(opt.maxnormout) -- affects params
      --print("update done")
      if opt.progress then
         xlua.progress(math.min(i + opt.seqlen, opt_trainsize), opt_trainsize)
      end
      
      if i%100==1  then --5000
            local n=1
            local decoded = decode(outputs,indexToChars,n)
            print('#','in:       ','label: ','out:   ','err: ')
            print(i,batch.inputWords[n],batch.labelWords[n],decoded,err)
            --[[print('activations')
            for t,z in pairs(outputs) do
                local str=''
                for ii=1, z:size(2) do
                    str = str..z[n][ii]..', '
                end
                print(t,str)
            end
            print('label')
            for t,z in pairs(targets) do
                local str=''
                for ii=1, z:size(2) do
                    str = str..z[n][ii]..', '
                end
                print(t,str)
            end
            print('grad')
            for t,z in pairs(gradOutputs) do
                local str=''
                for ii=1, z:size(2) do
                    str = str..z[n][ii]..', '
                end
                print(t,str)
            end
            
            --if (decoded=='') then
            --    assert(false)
            --end]]
            
      end

      if i % 1000 == 0 then
         collectgarbage()
      end

   end
   
   -- learning rate decay
   if opt.schedule then
      opt.lr = opt.schedule[epoch] or opt.lr
   else
      opt.lr = opt.lr + (opt.minlr - opt.startlr)/opt.saturate
   end
   opt.lr = math.max(opt.minlr, opt.lr)
   
   if not opt.silent then
      print("learning rate", opt.lr)
      if opt.meanNorm then
         print("mean gradParam norm", opt.meanNorm)
      end
   end

   if cutorch then cutorch.synchronize() end
   local speed = a:time().real/opt_trainsize
   print(string.format("Speed : %f sec/batch ", speed))

   local ppl = (sumErr/(opt_trainsize*opt.batchsize))
   print("Training error : "..ppl)

   xplog.trainppl[epoch] = ppl

   -- 2. cross-validation

   lm:evaluate()
   local sumErr = 0
   for i, batch in pairs(validSet) do
      --targets = targetmodule:forward(targets)
      local outputs = lm:forward(batch.inputs)
      local targets = batch.labels[1]
      local err = criterion:forward(outputs, targets)
      sumErr = sumErr+err
   end

   local ppl = (sumErr/(opt_validsize*opt.batchsize))
   print("Validation err : "..ppl)

   xplog.valppl[epoch] = ppl
   ntrial = ntrial + 1

   -- early-stopping
   if ppl < xplog.minvalppl then
      -- save best version of model
      xplog.minvalppl = ppl
      xplog.epoch = epoch 
      local filename = paths.concat(opt.savepath, opt.id..'.t7')
      print("Found new minima. Saving to "..filename)
      torch.save(filename, xplog)
      ntrial = 0
   elseif ntrial >= opt.earlystop then
      print("No new minima found after "..ntrial.." epochs.")
      print("Stopping experiment.")
      break
   end
    
   --print ('garb')
   collectgarbage()
   --print ('done garb')
   epoch = epoch + 1
end
print("Evaluate model using : ")
print("th evaluate-simple.lua --xplogpath "..paths.concat(opt.savepath, opt.id..'.t7')..(opt.cuda and '--cuda' or ''))
