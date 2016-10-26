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
1: syb_word_pos syb_line_pos <word vec> <syb vec>
2: syb_word_pos syb_line_pos <word vec> <syb vec>
...
LABEL (table of tensors at time slice of 1/(16*3)):
1: pitch_vec END (START)
]]

require 'paths'
local utf8 = require 'lua-utf8'

local lengthTable = {
    ['zero'] = 0,
    ['sixteenth'] = 3,
    ['half'] = 24,
    ['whole'] = 48,
    ['triplet-eighth'] = 4,
    ['eighth'] = 6,
    ['quarter'] = 12,
    ['triplet-quarter'] = 8
}

function getData(fileName,testSplit,batchSize,useBi)

    local inputVectorSize, inputs, labelVectorSize, labels = parseMusicFile(fileName,useBi)
    local trainInput,trainLabel,testInput,testLabel = splitShuffleData(inputs,labels,testSplitPortion)
    local trainBatches = make_batches(trainInput, trainLabels, batchSize)
    local testBatches = make_batches(testInput, testLabels, batchSize)

    return inputVectorSize, labelVectorSize, trainBatches, testBatches
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

function parseMusicFile(file, useBi)
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

    local inputVectorSize = 2+word_vec_len+syb_vec_len
    local labelVectorSize = max_pitch-min_pitch +3 --+1 for pitch range, +1 for rest, +1 for END token
    if useBi then
        labelVectorSize = labelVectorSize +1 --+1 for START token
    end
    local i=8
    local inTen=false
    local labelVecs={}
    local inputs = {}
    local labels = {}
    while  i<#lines do
        if string.len(lines[i])>0 then
            local syb, word, syllable_word_pos, num_syllables_in_word, syllable_line_pos, num_syllables_in_line, word_syb_vecS = string.match(lines[i],'(\w+)\s+(\w+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(.+)')
            if syb==start_token then
                if inTen ~= false then
                    table.insert(inputs,inTen)
                    inTen=false
                end
                if #labelVecs>0 then
                    table.insert(labels,labelVecs)
                    labelTens = {}
                end
            end 
            local inVec = torch.Tensor(1,inputVectorSize)
            inVec[1][1] = tonumber(syllable_word_pos)/tonumber(num_syllables_in_word)
            inVec[1][2] = tonumber(syllable_line_pos)/tonumber(num_syllables_in_line)
            local j=3
            for w in word_syb_vecS:gmatch("%S+") do
                inVec[i][j]=tonumber(w)
                j=j+1
            end
            assert(j-1==inputVectorSize)
            if inTen==false then
                inTen = inVec
            else
                inTen = torch.cat(inTen,inVec,1)
            end

            local begin = #labelVecs
            while i<#lines do
                local pitch, lengthS, measure, beat = string.match(lines[i],'(-?\d+)\s+(\w+)\s+(\d+)\s+(\d*.?\d+)')
                if pitch == stop_pitch then
                    break
                end

                local labelVec = torch.zeros(1,labelVectorSize)
                if pitch ~= rest_pitch then
                    labelVec[1][pitch-minPitch+1]=1
                else
                    labelVec[1][max_pitch-minPitch+2]=1
                end

                local length = lengthTable[lengthS]
                for j=1, length do
                    table.insert(labelVecs,labelVec)
                end
                i = i+1
            end
            if useBi then
                labelVecs[begin+1] = labelVecs[begin+1]:clone()
                labelVecs[begin+1][1][labelVectorSize]=1 --START token
            end
            labelVecs[#labelVecs] = labelVecs[#labelVecs]:clone()
            labelVecs[#labelVecs][1][max_pitch-min_pitch +2]=1 --END token

        end
        i = i+1
    end
    table.insert(inputs,inTen)
    table.insert(labels,labelVecs)

    return inputVectorSize, inputs, labelVectorSize, labels
end 

function splitShuffleData(inputs,labels,testSplitPortion)

    local splitAt = math.floor(testSplitPortion*#inputs)
    local trainInputs = {}
    local trainLabels = {}
    local testInputs = {}
    local testLabels = {}

    for i = 1, splitAt-1 do
        table.insert(testInputs,inputs[i])
        table.insert(testLabels,labels[i])
    end
    for i = splitAt, #inputs do
        table.insert(trainInputs,inputs[i])
        table.insert(trainLabels,labels[i])
    end
    shuffleTables(trainInputs,trainLabels)
    shuffleTables(testInputs,testLabels)

    return trainInputs,trainLabels,testInputs,testLabels
end



--This both breaks the tables into batches, and formats them into what warp-ctc expects
function make_batches(inputVectors, labelTables, batchSize)
    local batchesRet={}
    --local labelRet={}
    
    local batchInput={}
    local batchLabel={}
    local maxLength=0
    for i=1, #inputVectors do
    --for i=1, math.min(100,#inputVectors) do
        table.insert(batchInput,inputVectors[i])
        table.insert(batchLabel,labelTables[i])
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
            maxLength=0
        end
    end
    
    return batchesRet
end


