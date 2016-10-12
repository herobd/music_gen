--[[
File fomat:

word_vec_len: #
syb_vec_len: #
pitch_range: low# high#
rest_pitch: #
stop_token_pitch: #
start_token: <str>
note_values: <space seperated list of strs>

<syl> <word> #syllable_word_pos  #num_syllables_in_word  #syllable_line_pos  #num_syllables_in_line <list: word vec> <list: syl vec>
\t  #pitch <note_val> #measure #beat_in_measure
\t  ....
\t  stop_token_pitch zero ...

Each sequence is begun with <start_token> syl/word with some rest notes

Parsed data should look like:
IN (table of tensors):
1: <word vec>+<syb vec>
2: <word vec>+<syb vec>
...
LABEL (table of tensors):
1: pitch
]]

require 'paths'
local utf8 = require 'lua-utf8'
local dl = require 'dataload'

version = 0.1
useBi=false

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

function parseMusicFile(file)
    local lines = lines_from(file)
    local word_vec_len = tonumber(string.match(lines[1],'.*: (\d+)'))
    local syb_vec_len = tonumber(string.match(lines[2],'.*: (\d+)'))
    local min_pitch
    local max_pitch
    min_pitch, max_pitch = tonumber(string.match(lines[3],'.*: (\d+) (\d+)'))
    local rest_pitch = tonumber(string.match(lines[4],'.*: (\d+)'))
    local stop_pitch = tonumber(string.match(lines[5],'.*: (\d+)'))
    local start_token = (string.match(lines[6],'.*: (.+)'))
    local note_valuesS = (string.match(lines[7],'.*: (.+)'))
    local note_values={}
    for i in string.gmatch(not_valuesS, "%S+") do
        table.insert(note_values,i)
    end

    for i=8, #lines do

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
        local labelTensorSize=labelCounter --this has the +1 for STOP token
        if useBi then
            labelTensorSize=labelTensorSize+1 --for START token
        end
        for t =1, utf8.len(labelWords[i]) do
            local labelTensor = torch.Tensor(1,labelTensorSize):zero()
            local uc=utf8.sub(labelWords[i],t,t)
            local c=utf8.lower(uc)
            labelTable[t]=labelChars[c]
            labelTensor[1][labelChars[c]]=1
            if isUpper(uc) then
                labelTensor[1][labelCounter]=1
            end

            if useBi and lastChar~=c then --this is a start char
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
        local labelTensorSize=labelCounter --this has the +1 for STOP token
        if useBi then
            labelTensorSize=labelTensorSize+1 --for START token
        end
        for t =1, utf8.len(labelWords[i]) do
            local labelTensor = torch.Tensor(1,labelTensorSize):zero()
            local uc=utf8.sub(labelWords[i],t,t)
            local c=utf8.lower(uc)
            labelTable[t]=labelChars[c]
            labelTensor[1][labelChars[c]]=1
            if isUpper(uc) then
                labelTensor[1][labelCounter]=1
            end
            if useBi and lastChar~=c then --this is a startChar
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
        acts[#indexToChars +1]=0;
        local str=false
        if useBi then
            str = acts[#indexToChars +2]>0.5
            acts[#indexToChars +2]=0;
        end
        local maxs, indices = torch.max(acts,1)
        local c = indexToChars[ indices[1] ]
        if cap then
            c=utf8.upper(c)
        end
        if useBi and str then
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

cmd:option('--bi', false, 'use Bidirectional') 

cmd:text()
local opt = cmd:parse(arg or {})
opt.hiddensize = loadstring(" return "..opt.hiddensize)()
opt.schedule = loadstring(" return "..opt.schedule)()
if not opt.silent then
   table.print(opt)
end
opt.id = opt.id == '' and ('ptb' .. ':' .. dl.uniqueid()) or opt.id
useBi = opt.bi
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
