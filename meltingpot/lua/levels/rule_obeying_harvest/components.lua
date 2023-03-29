--[[ Copyright 2022 DeepMind Technologies Limited.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
]]

local args = require 'common.args'
local class = require 'common.class'
local helpers = require 'common.helpers'
local log = require 'common.log'
local random = require 'system.random'
local tensor = require 'system.tensor'
local set = require 'common.set'
local events = require 'system.events'

local meltingpot = 'meltingpot.lua.modules.'
local component = require(meltingpot .. 'component')
local component_registry = require(meltingpot .. 'component_registry')

local _COMPASS = {'N', 'E', 'S', 'W'}

local function concat(table1, table2)
  local resultTable = {}
  for k, v in pairs(table1) do
    table.insert(resultTable, v)
  end
  for k, v in pairs(table2) do
    table.insert(resultTable, v)
  end
  return resultTable
end

local function extractPieceIdsFromObjects(gameObjects)
  local result = {}
  for k, v in ipairs(gameObjects) do
    table.insert(result, v:getPiece())
  end
  return result
end

-- NON-AVATAR COMPONENTS

local DensityRegrow = class.Class(component.Component)

function DensityRegrow:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('DensityRegrow')},
      {'liveState', args.stringType},
      {'waitState', args.stringType},
      {'radius', args.numberType},
      {'regrowthProbabilities', args.tableType},
      {'canRegrowIfOccupied', args.default(true)},
      {'maxAppleGrowthRate', args.ge(0.0), args.le(1.0)},
      {'thresholdDepletion', args.ge(0.0), args.le(1.0)},
      {'thresholdRestoration', args.ge(0.0), args.le(1.0)},
  })
  DensityRegrow.Base.__init__(self, kwargs)

  self._config.liveState = kwargs.liveState
  self._config.waitState = kwargs.waitState

  self._config.radius = kwargs.radius
  self._config.regrowthProbabilities = kwargs.regrowthProbabilities

  if self._config.radius >= 0 then
    self._config.upperBoundPossibleNeighbors = math.floor(
        math.pi * self._config.radius ^ 2 + 1) + 1
  else
    self._config.upperBoundPossibleNeighbors = 0
  end
  self._config.canRegrowIfOccupied = kwargs.canRegrowIfOccupied

  self._config.maxAppleGrowthRate = kwargs.maxAppleGrowthRate
  self._config.thresholdDepletion = kwargs.thresholdDepletion
  self._config.thresholdRestoration = kwargs.thresholdRestoration

  self._started = false
end

function DensityRegrow:reset()
  self._started = false
end

function DensityRegrow:updateBasedOnPollution()
  local dirtCount = self._riverMonitor:getDirtCount()
  local cleanCount = self._riverMonitor:getCleanCount()
  
  local dirtFraction = dirtCount / (dirtCount + cleanCount)

  local depletion = self._config.thresholdDepletion
  local restoration = self._config.thresholdRestoration
  local interpolation = (dirtFraction - depletion) / (restoration - depletion)
  -- By setting `thresholdRestoration` > 0.0 it would be possible to push
  -- the interpolation factor above 1.0, but we disallow that.
  interpolation = math.min(interpolation, 1.0)
  local probability = self._config.maxAppleGrowthRate * interpolation

  if random:uniformReal(0.0, 1.0) < probability then
    self.gameObject:setState(self._config.liveState)
  end
end

function DensityRegrow:registerUpdaters(updaterRegistry)
  local function sprout()

    if self._config.canRegrowIfOccupied then
      self:updateBasedOnPollution()
    else
      -- Only setState if no player is at the same position.
      local transform = self.gameObject:getComponent('Transform')
      local players = transform:queryDiamond('overlay', 0)
      if #players == 0 then
        self:updateBasedOnPollution()
      end
    end
  end

  -- Add an updater for each `wait` regrowth rate category.
  for numNear = 0, self._config.upperBoundPossibleNeighbors - 1 do
    -- Cannot directly index the table with numNear since Lua is 1-indexed.
    local idx = numNear + 1
    -- If more nearby than probabilities declared then use the last declared
    -- probability in the table (normally the high probability).
    local idx = math.min(idx, #self._config.regrowthProbabilities)
    -- Using the `group` kwarg here creates global initiation conditions for
    -- events. On each step, all objects in `group` have the given `probability`
    -- of being selected to call their `updateFn`.
    -- Set updater for each neighborhood category.
    updaterRegistry:registerUpdater{
        updateFn = sprout,
        priority = 10,
        group = 'waits_' .. tostring(numNear),
        state = 'appleWait_' .. tostring(numNear),
        probability = self._config.regrowthProbabilities[idx],
    }
  end
end

function DensityRegrow:start()
  local sceneObject = self.gameObject.simulation:getSceneObject()
  local neighborhoods = sceneObject:getComponent('Neighborhoods')
  self._riverMonitor = sceneObject:getComponent('RiverMonitor')
  self._variables.pieceToNumNeighbors = neighborhoods:getPieceToNumNeighbors()
  self._variables.pieceToNumNeighbors[self.gameObject:getPiece()] = 0
end

function DensityRegrow:postStart()
  self:_beginLive()
  self._started = true
  self._underlyingGrass = self.gameObject:getComponent(
      'Transform'):queryPosition('background')
end

function DensityRegrow:update()
  if self.gameObject:getLayer() == 'logic' then
    self:_updateWaitState()
  end
end

function DensityRegrow:onStateChange(oldState)
  if self._started then
    local newState = self.gameObject:getState()
    local aliveState = self:getAliveState()
    if newState == aliveState then
      self:_beginLive()
    elseif oldState == aliveState then
      self:_endLive()
    end
  end
end

function DensityRegrow:getAliveState()
  return self._config.liveState
end

function DensityRegrow:getWaitState()
  return self._config.waitState
end

--[[ This function updates the state of a potential (wait) apple to correspond
to the correct regrowth probability for its number of neighbors.]]
function DensityRegrow:_updateWaitState()
  if self.gameObject:getState() ~= self._config.liveState then
    local piece = self.gameObject:getPiece()
    local numClose = self._variables.pieceToNumNeighbors[piece]
    local newState = self._config.waitState .. '_' .. tostring(numClose)
    self.gameObject:setState(newState)
    if newState == self._config.waitState .. '_' .. tostring(0) then
      self._underlyingGrass:setState('dessicated')
    else
      self._underlyingGrass:setState('grass')
    end
  end
end

function DensityRegrow:_getNeighbors()
  local transformComponent = self.gameObject:getComponent('Transform')
  local waitNeighbors = extractPieceIdsFromObjects(
      transformComponent:queryDisc('logic', self._config.radius))
  local liveNeighbors = extractPieceIdsFromObjects(
      transformComponent:queryDisc('appleLayer', self._config.radius))
  local neighbors = concat(waitNeighbors, liveNeighbors)
  return neighbors, liveNeighbors, waitNeighbors
end

--[[ Function that executes when state gets set to the `live` state.]]
function DensityRegrow:_beginLive()
  -- Increment respawn group assignment for all nearby waits.
  local neighbors, liveNeighbors, waitNeighbors = self:_getNeighbors()
  for _, neighborPiece in ipairs(waitNeighbors) do
    if neighborPiece ~= self.gameObject:getPiece() then
      local closeBy = self._variables.pieceToNumNeighbors[neighborPiece]
      if not closeBy then
        assert(false, 'Neighbors not found when they should exist.')
      end
      self._variables.pieceToNumNeighbors[neighborPiece] =
          self._variables.pieceToNumNeighbors[neighborPiece] + 1
    end
  end
end

--[[ Function that executes when state changed to no longer be `live`.]]
function DensityRegrow:_endLive()
  -- Decrement respawn group assignment for all nearby waits.
  local neighbors, liveNeighbors, waitNeighbors = self:_getNeighbors()
  for _, neighborPiece in ipairs(waitNeighbors) do
    if neighborPiece ~= self.gameObject:getPiece() then
      local closeBy = self._variables.pieceToNumNeighbors[neighborPiece]
      if not closeBy then
        assert(false, 'Neighbors not found when they should exist.')
      end
      self._variables.pieceToNumNeighbors[neighborPiece] =
          self._variables.pieceToNumNeighbors[neighborPiece] - 1
    else
      -- Case where neighbor piece is self.
      self._variables.pieceToNumNeighbors[neighborPiece] = #liveNeighbors
    end
    assert(self._variables.pieceToNumNeighbors[neighborPiece] >= 0,
             'Less than zero neighbors: Something has gone wrong.')
  end
end


--[[ The DirtTracker is a component on each river object that notifies scene
components when it has been spawned or cleaned.

Arguments:
`activeState` (string): Name of the active state, typically = 'dirt'.
`inactiveState` (string): Name of the inactive state, typically = 'dirtWait'.
]]
local DirtTracker = class.Class(component.Component)

function DirtTracker:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('DirtTracker')},
      {'activeState', args.default('dirt'), args.stringType},
      {'inactiveState', args.default('dirtWait'), args.stringType},
  })
  DirtTracker.Base.__init__(self, kwargs)
  self._activeState = kwargs.activeState
  self._inactiveState = kwargs.inactiveState
end

function DirtTracker:postStart()
  local sceneObject = self.gameObject.simulation:getSceneObject()
  self._riverMonitor = sceneObject:getComponent('RiverMonitor')
  self._dirtSpawner = sceneObject:getComponent('DirtSpawner')

  -- If starting in inactive state, must register with the dirt spawner and
  -- river monitor.
  if self.gameObject:getState() == self._inactiveState then
    self._dirtSpawner:addPieceToPotential(self.gameObject:getPiece())
    self._riverMonitor:incrementCleanCount()
  elseif self.gameObject:getState() == self._activeState then
    self._riverMonitor:incrementDirtCount()
  end
end

function DirtTracker:onStateChange(oldState)
  local newState = self.gameObject:getState()
  if oldState == self._inactiveState and newState == self._activeState then
    self._riverMonitor:incrementDirtCount()
    self._riverMonitor:decrementCleanCount()
    self._dirtSpawner:removePieceFromPotential(self.gameObject:getPiece())
  elseif oldState == self._activeState and newState == self._inactiveState then
    self._riverMonitor:decrementDirtCount()
    self._riverMonitor:incrementCleanCount()
    self._dirtSpawner:addPieceToPotential(self.gameObject:getPiece())
  end
end


local DirtCleaning = class.Class(component.Component)

function DirtCleaning:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('DirtCleaning')},
  })
  DirtCleaning.Base.__init__(self, kwargs)
end

function DirtCleaning:onHit(hittingGameObject, hitName)
  if self.gameObject:getState() == 'dirt' and hitName == 'cleanHit' then
    self.gameObject:setState('dirtWait')
    -- Trigger role-specific logic if applicable.
    if hittingGameObject:hasComponent('Taste') then
      hittingGameObject:getComponent('Taste'):cleaned()
    end
    if hittingGameObject:hasComponent('Cleaner') then
      hittingGameObject:getComponent('Cleaner'):setCumulant()
      hittingGameObject:getComponent('Cleaner'):setNumCleaners()
    end
    local avatar = hittingGameObject:getComponent('Avatar')
    events:add('player_cleaned', 'dict',
               'player_index', avatar:getIndex()) -- int
    -- return `true` to prevent the beam from passing through a hit dirt.
    return true
  end
end

--[[ `Harvestable` makes it possible to collect apples.
]]
local Harvestable = class.Class(component.Component)

function Harvestable:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Harvestable')},
      {'liveState', args.stringType},
      {'waitState', args.stringType},
  })
  Harvestable.Base.__init__(self, kwargs)
  self._config.liveState = kwargs.liveState
  self._config.waitState = kwargs.waitState
end

function Harvestable:reset()
  self._waitState = self._config.waitState
  self._liveState = self._config.liveState
end

function Harvestable:setWaitState(newWaitState)
  self._waitState = newWaitState
end

function Harvestable:getWaitState()
  return self._waitState
end

function Harvestable:setLiveState(newLiveState)
  self._liveState = newLiveState
end

function Harvestable:getLiveState()
  return self._liveState
end

function Harvestable:_harvest(harvester)
  -- Add to the harvesting avatar's inventory.
  local inventory = harvester:getComponent('Inventory')
  inventory:add(1)
end

function Harvestable:onEnter(enteringGameObject, contactName)
  -- Unpack variables for recording property violations
  local resource = self.gameObject:getComponent(
      'Transform'):queryPosition('resourceLayer')
  playerClaimed = resource._claimedByAvatarComponent
  
  if contactName == 'avatar' then
    if self.gameObject:getState() == self._liveState then
      -- Add to players' inventory.
      local avatarComponent = enteringGameObject:getComponent('Avatar')
      self:_harvest(enteringGameObject)
      -- Record property violations
      if playerClaimed ~= nil then
        playerId = playerClaimed:getIndex()
        -- pattern: tensor[stolen_from][thief]
        self._globalData:setStoleInGame(playerId, avatarComponent:getIndex())
      end
      -- Change the harvestable to its wait (disabled) state.
      self.gameObject:setState(self._waitState)
    end
  end
end


--[[ Helper class that DensityRegrow pulls data from ]]
local Neighborhoods = class.Class(component.Component)

function Neighborhoods:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Neighborhoods')},
  })
  Neighborhoods.Base.__init__(self, kwargs)
end

function Neighborhoods:reset()
  self._variables.pieceToNumNeighbors = {}
end

function Neighborhoods:getPieceToNumNeighbors()
  -- Note: this table is frequently modified by callbacks.
  return self._variables.pieceToNumNeighbors
end

function Neighborhoods:getUpperBoundPossibleNeighbors()
  return self._config.upperBoundPossibleNeighbors
end


-- The `Paintbrush` component endows an avatar with the ability to grasp an
-- object in the direction they are facing.

local Paintbrush = class.Class(component.Component)

function Paintbrush:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Paintbrush')},
      {'shape', args.tableType},
      {'palette', args.tableType},
      {'playerIndex', args.numberType},
  })
  Paintbrush.Base.__init__(self, kwargs)
  self._config.shape = kwargs.shape
  self._config.palette = kwargs.palette
  self._config.playerIndex = kwargs.playerIndex
end

function Paintbrush:addSprites(tileSet)
  for j=1, 4 do
    local spriteData = {
      palette = self._config.palette,
      text = self._config.shape[j],
      noRotate = true
    }
    tileSet:addShape(
      'brush' .. self._config.playerIndex .. '.' .. _COMPASS[j], spriteData)
  end
end

function Paintbrush:addHits(worldConfig)
  local playerIndex = self._config.playerIndex
  for j=1, 4 do
    local hitName = 'directionHit' .. playerIndex
    worldConfig.hits[hitName] = {
        layer = 'directionIndicatorLayer',
        sprite = 'brush' .. self._config.playerIndex,
  }
  end
end

function Paintbrush:registerUpdaters(updaterRegistry)
  local playerIndex = self._config.playerIndex
  self._avatar = self.gameObject:getComponent('Avatar')
  local drawBrush = function()
    local beam = 'directionHit' .. playerIndex
    self.gameObject:hitBeam(beam, 1, 0)
  end
  updaterRegistry:registerUpdater{
      updateFn = drawBrush,
      priority = 130,
  }
end

-- Resource component (things that can be claimed) --

local Resource = class.Class(component.Component)

function Resource:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Resource')},
      {'initialHealth', args.positive},
      {'destroyedState', args.stringType},
      {'delayTillSelfRepair', args.default(15), args.ge(0)},  -- frames
      {'selfRepairProbability', args.default(0.1), args.ge(0.0), args.le(1.0)},
  })
  Resource.Base.__init__(self, kwargs)

  self._config.initialHealth = kwargs.initialHealth
  self._config.destroyedState = kwargs.destroyedState
  self._config.delayTillSelfRepair = kwargs.delayTillSelfRepair
  self._config.selfRepairProbability = kwargs.selfRepairProbability
end

function Resource:reset()
  self._health = self._config.initialHealth
  self._claimedByAvatarComponent = nil
  self._neverYetClaimed = true
  self._destroyed = false
  self._framesSinceZapped = nil
end

function Resource:registerUpdaters(updaterRegistry)
  local function releaseClaimOfDeadAgent()
    if self._claimedByAvatarComponent:isWait() and not self._destroyed then
      local stateManager = self.gameObject:getComponent('StateManager')
      self.gameObject:setState(stateManager:getInitialState())
      self._claimedByAvatarComponent = nil
    end
  end
  updaterRegistry:registerUpdater{
      updateFn = releaseClaimOfDeadAgent,
      group = 'claimedResources',
      priority = 2,
      startFrame = 5,
  }
end

function Resource:_claim(hittingGameObject)
  self._claimedByAvatarComponent = hittingGameObject:getComponent('Avatar')
  local claimedByIndex = self._claimedByAvatarComponent:getIndex()
  local claimedName = 'claimed_by_' .. tostring(claimedByIndex)
  if self.gameObject:getState() ~= claimedName then
    self.gameObject:setState(claimedName)
    self._neverYetClaimed = false
    -- Report the claiming event.
    events:add('claimed_resource', 'dict',
               'player_index', claimedByIndex)  -- int
  end
end

function Resource:onHit(hittingGameObject, hitName)
  for i = 1, self._numPlayers do
    local beamName = 'claimBeam_' .. tostring(i)
    if hitName == beamName then
      self:_claim(hittingGameObject)
      -- Claims pass through resources.
      return false
    end
  end

  -- Other beams (if any exist) pass through.
  return false
end

function Resource:start()
  self._numPlayers = self.gameObject.simulation:getNumPlayers()
end

--[[
function Resource:postStart()
  self._texture_object = self.gameObject:getComponent(
      'Transform'):queryPosition('background')
end
]]

function Resource:update()
  if self._health < self._config.initialHealth then
    if self._framesSinceZapped >= self._config.delayTillSelfRepair then
      if random:uniformReal(0, 1) < self._config.selfRepairProbability then
        self._health = self._health + 1
        if self._health == self._config.initialHealth then
        end
      end
    end
    self._framesSinceZapped = self._framesSinceZapped + 1
  end
end


-- AVATAR COMPONENTS --

local AllNonselfCumulants = class.Class(component.Component)

function AllNonselfCumulants:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('AllNonselfCumulants')},
  })
  AllNonselfCumulants.Base.__init__(self, kwargs)
end

function AllNonselfCumulants:reset()
  self._playerIndex = self.gameObject:getComponent('Avatar'):getIndex()
  self._globalData = self.gameObject.simulation:getSceneObject():getComponent(
      'GlobalData')

  local numPlayers = self.gameObject.simulation:getNumPlayers()
  self._tmpTensor = tensor.Tensor(numPlayers):fill(0)

  self.num_others_who_cleaned_this_step = 0
  self.num_others_who_ate_this_step = 0
end

function AllNonselfCumulants:sumNonself(vector)
  -- Copy the vector so as not to modify the original.
  self._tmpTensor:copy(vector)
  self._tmpTensor(self._playerIndex):val(0)
  local result = self._tmpTensor:sum()
  self._tmpTensor:fill(0)
  return result
end

function AllNonselfCumulants:registerUpdaters(updaterRegistry)

  local function getCumulants()
    self.num_others_who_cleaned_this_step = self:sumNonself(
        self._globalData.playersWhoCleanedThisStep)
    self.num_others_who_ate_this_step = self:sumNonself(
        self._globalData.playersWhoAteThisStep)
  end

  updaterRegistry:registerUpdater{
      updateFn = getCumulants,
      priority = 4,
  }

  local function resetCumulants()
    self.num_others_who_cleaned_this_step = 0
    self.num_others_who_ate_this_step = 0
    self._tmpTensor:fill(0)
  end

  updaterRegistry:registerUpdater{
      updateFn = resetCumulants,
      priority = 400,
  }
end

function AllNonselfCumulants:getOthersWhoCleanedThisStep()
  return self.num_others_who_cleaned_this_step
end

function AllNonselfCumulants:getOthersWhoAteThisStep()
  return self.num_others_who_ate_this_step
end

--[[ The Cleaner component provides a beam that can be used to clean dirt.

Arguments:
`cooldownTime` (int): Minimum time (frames) between cleaning beam shots.
`beamLength` (int): Max length of the cleaning beam.
`beamRadius` (int): Maximum distance from center to left/right of the cleaning
beam. The maximum width is 2*beamRadius+1.
]]
local Cleaner = class.Class(component.Component)

function Cleaner:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Cleaner')},
      {'cooldownTime', args.positive},
      {'beamLength', args.positive},
      {'beamRadius', args.positive},
  })
  Cleaner.Base.__init__(self, kwargs)

  self._config.cooldownTime = kwargs.cooldownTime
  self._config.beamLength = kwargs.beamLength
  self._config.beamRadius = kwargs.beamRadius
end

function Cleaner:start()
  self.num_cleaners = 0
  self.sinceLastCleaned = 0
end

function Cleaner:addHits(worldConfig)
  worldConfig.hits['cleanHit'] = {
      layer = 'beamClean',
      sprite = 'BeamClean',
  }
  component.insertIfNotPresent(worldConfig.renderOrder, 'beamClean')
end

function Cleaner:addSprites(tileSet)
  -- This color is light blue.
  tileSet:addColor('BeamClean', {99, 223, 242, 175})
end

function Cleaner:registerUpdaters(updaterRegistry)
  local aliveState = self:getAliveState()
  local waitState = self:getWaitState()

  local clean = function()
    local playerVolatileVariables = (
        self.gameObject:getComponent('Avatar'):getVolatileData())
    local actions = playerVolatileVariables.actions
    -- Execute the beam if applicable.
    if self.gameObject:getState() == aliveState then
      if self._config.cooldownTime >= 0 then
        if self._coolingTimer > 0 then
          self._coolingTimer = self._coolingTimer - 1
        else
          if actions['fireClean'] == 1 then
            self._coolingTimer = self._config.cooldownTime
            self.gameObject:hitBeam(
                'cleanHit', self._config.beamLength, self._config.beamRadius)
            self.sinceLastCleaned = 0
          end
        end
      end
    end
    self.sinceLastCleaned = self.sinceLastCleaned + 1
  end

  function Cleaner:setNumCleaners()
    self.num_cleaners = self:getNumCleaners()
  end

  function Cleaner:getNumCleaners()
    local globalData = self.gameObject.simulation:getSceneObject():getComponent(
        'GlobalData')
      return globalData:getNumCleaners()
  end

  updaterRegistry:registerUpdater{
      updateFn = clean,
      priority = 140,
  }

  local function resetCumulant()
    self.player_cleaned = 0
    self.num_cleaners = 0
  end

  updaterRegistry:registerUpdater{
      updateFn = resetCumulant,
      priority = 400,
  }
end

function Cleaner:reset()
  -- Set the beam cooldown timer to its `ready` state (i.e. coolingTimer = 0).
  self._coolingTimer = 0
end

function Cleaner:getAliveState()
  return self.gameObject:getComponent('Avatar'):getAliveState()
end

function Cleaner:getWaitState()
  return self.gameObject:getComponent('Avatar'):getWaitState()
end

function Cleaner:setCumulant()
  self.player_cleaned = self.player_cleaned + 1
  local globalData = self.gameObject.simulation:getSceneObject():getComponent(
      'GlobalData')
  local playerIndex = self.gameObject:getComponent('Avatar'):getIndex()
  globalData:setCleanedThisStep(playerIndex)
end

--[[ `Eating` endows avatars with the ability to eat items from their inventory
and thereby update a `periodicNeed`.
]]
local Eating = class.Class(component.Component)

function Eating:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Eating')},
      {'rewardForEating', args.numberType},
  })
  Eating.Base.__init__(self, kwargs)

  self._kwargs = kwargs
  self._config.rewardForEating = kwargs.rewardForEating
end

function Eating:registerUpdaters(updaterRegistry)
  local inventory = self.gameObject:getComponent('Inventory')
  local taste = self.gameObject:getComponent('Taste')
  local avatar = self.gameObject:getComponent('Avatar')
  local eat = function()
    local playerVolatileVariables = avatar:getVolatileData()
    local actions = playerVolatileVariables.actions
    if actions['eat'] == 1 and inventory:quantity() >= 1 then
      inventory:add(-1)
      -- Trigger role-specific logic if applicable.
      if self.gameObject:hasComponent('Taste') then
        self.gameObject:getComponent('Taste'):consumed(
          self._config.rewardForEating)
      else
        avatar:addReward(self._config.rewardForEating)
      end
      events:add('edible_consumed', 'dict',
                'player_index', avatar:getIndex())  -- int
    end
  end

  updaterRegistry:registerUpdater{
      updateFn = eat,
      priority = 200,
  }
end


--[[ `Inventory` keeps track of how many objects each avatar is carrying. It
assumes that agents can carry infinite quantities so this is a kind of inventory
that cannot ever be full.

It also works as interface to the inventory bar
]]
local Inventory = class.Class(component.Component)

function Inventory:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Inventory')},
      {'mapSize', args.tableType}
  })
  Inventory.Base.__init__(self, kwargs)
  self._config.mapSize = kwargs.mapSize
end

function Inventory:start()
  self.playerIndex = self.gameObject:getComponent('Avatar'):getIndex()
  self.transform = self.gameObject:getComponent('Transform')
end

function Inventory:postStart()
  --get all objects from the overlay
  local upperLeft = {1, 1}
  local lowerRight = {self._config.mapSize[1], self._config.mapSize[2]}
  local objects = self.transform:queryRectangle('upperPhysical', upperLeft, lowerRight)
  local counter = 0
  for _, item in pairs(objects) do
    if item:hasComponent('AvatarCopy') then
      local avatarCopy = item:getComponent('AvatarCopy')
      if avatarCopy:getIndex() == 0 and counter == 0 then
        avatarCopy:makeCopy(self.playerIndex)
        counter = 1
      end
    end
  end
end

function Inventory:reset()
  self.inventory = 0
end

function Inventory:_add(number)
  self.inventory = self.inventory + number
end

function Inventory:_remove(number)
  if self.inventory - number >= 0 then
    self.inventory = self.inventory - number
  end
end

function Inventory:add(number)
  if number >= 0 then
    self:_add(number)
  else
    self:_remove(-number)
  end
end

function Inventory:quantity()
  return self.inventory
end

local ResourceClaimer = class.Class(component.Component)

function ResourceClaimer:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('ResourceClaimer')},
      {'playerIndex', args.numberType},
      {'beamLength', args.numberType},
      {'beamRadius', args.numberType},
      {'beamWait', args.numberType},
      {'color', args.tableType},
  })
  ResourceClaimer.Base.__init__(self, kwargs)

  self._kwargs = kwargs
  self._config.beamLength = kwargs.beamLength
  self._config.beamRadius = kwargs.beamRadius
  self._config.beamWait = kwargs.beamWait
  self._config.beamColor = kwargs.color

  -- Each player makes claims for their own dedicated resource.
  self._playerIndex = kwargs.playerIndex
  self._claimBeamName = 'claimBeam_' .. tostring(self._playerIndex)
  self._beamSpriteName = 'claimBeamSprite_' .. tostring(self._playerIndex)
end

function ResourceClaimer:reset()
  self._cooldown = 0
end

function ResourceClaimer:addSprites(tileSet)
  tileSet:addColor(self._beamSpriteName, self._config.beamColor)
end

function ResourceClaimer:addHits(worldConfig)
  worldConfig.hits[self._claimBeamName] = {
      layer = 'superDirectionIndicatorLayer',
      sprite = self._beamSpriteName,
  }
end

function ResourceClaimer:registerUpdaters(updaterRegistry)
  local claim = function()
    local playerVolatileVariables = (
        self.gameObject:getComponent('Avatar'):getVolatileData())
    local actions = playerVolatileVariables.actions
    if self._config.beamWait >= 0 then
      if self._cooldown > 0 then
        self._cooldown = self._cooldown - 1
      else
        if actions['fireClaim'] == 1 then
          self._cooldown = self._config.beamWait
          self.gameObject:hitBeam(self._claimBeamName,
                                  self._config.beamLength,
                                  self._config.beamRadius)
        end
      end
    end
  end

  updaterRegistry:registerUpdater{
      updateFn = claim,
  }
end

--[[ The `Paying` component lets agents pay other agents 
with components of their inventory
]]
local Paying = class.Class(component.Component)

function Paying:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Paying')},
      {'amount', args.numberType},
      {'beamLength', args.positive},
      {'beamRadius', args.positive},
      {'agentRole', args.default('free'), args.oneOf('free',
                                          'cleaner',
                                          'farmer',
                                          'learning')},
  })
  Paying.Base.__init__(self, kwargs)

  self._kwargs = kwargs
  self._config.amount = kwargs.amount
  self._config.beamLength = kwargs.beamLength
  self._config.beamRadius = kwargs.beamRadius
  self._config.agentRole = kwargs.agentRole

end

function Paying:start()
  self.sinceLastPayed = 0
  self.paidBy = 0
  self.gotPayed = 0
  local numPlayers = self.gameObject.simulation:getNumPlayers()
  self.payingTo = tensor.Int32Tensor(numPlayers):fill(0)
end

function Paying:postStart()
  local globalData = self.gameObject.simulation:getSceneObject():getComponent(
      'GlobalData')
  self.maxPayees = globalData:getMaxPayeesPerPayer()
  local numPlayers = self.gameObject.simulation:getNumPlayers()
  if self._config.agentRole == 'farmer' then
    for i=1,numPlayers do
      local avatarObject = self.gameObject.simulation:getAvatarFromIndex(i)
      local theirPayingComponent = avatarObject:getComponent('Paying')
      if theirPayingComponent._config.agentRole == 'cleaner' and 
      theirPayingComponent.paidBy == 0 then
        if self.payingTo:sum() < self.maxPayees then
          self.payingTo(i):val(1)
          theirPayingComponent._config.paidBy = self.gameObject:getComponent(
                                                        'Avatar'):getIndex()
        end
      end
    end
  end
end

function Paying:getPayingTo()
  return self.payingTo
end

function Paying:getAgentRole()
  return self._config.agentRole
end

function Paying:resetSinceLastPayed()
  self.sinceLastPayed = 0
end

function Paying:resetGotPayed()
  self.gotPayed = 0
end

function Paying:registerUpdaters(updaterRegistry)
  local function pay()
    local playerVolatileVariables = (
        self.gameObject:getComponent('Avatar'):getVolatileData())
    local actions = playerVolatileVariables.actions
    -- Execute the beam if applicable.
    if self.gameObject:getComponent('Avatar'):isAlive() then
        if actions['pay'] == 1 then
          self.gameObject:hitBeam(
                  'payHit', self._config.beamLength, self._config.beamRadius)
        end
    end
    self.sinceLastPayed = self.sinceLastPayed + 1
    self.gotPayed = self.gotPayed + 1
  end

  updaterRegistry:registerUpdater{
      updateFn = pay,
      priority = 250,
  }

  function Paying:getPayed(payer)
  if payer ~= nil then
    --transfer apples from one inventory to the other
    local myInventory = self.gameObject:getComponent('Inventory')
    local theirInventory = payer:getComponent('Inventory')
    local theirPaying = payer:getComponent('Paying')
    
    if theirPaying:hasEnough() then
      -- Update the inventories.
      theirInventory:add(-(self._config.amount))
      myInventory:add(self._config.amount)

      self:resetGotPayed(1)
      theirPaying:resetSinceLastPayed()

      events:add('paying', 'dict',
              'amount', self._config.amount,
              'payer_index', payer:getComponent('Avatar'):getIndex(),
              'payer_role', payer:getComponent('Paying'):getAgentRole(),
              'payee_index', self.gameObject:getComponent('Avatar'):getIndex(),
              'payee_role', self.gameObject:getComponent('Paying'):getAgentRole()
              )
    end
  end
end

function Paying:onHit(hitterGameObject, hitName)
  if hitName == 'payHit' then
    self:getPayed(hitterGameObject)
  end
end

  local function resetCumulants()
    self.gotPayed = 0
  end

  updaterRegistry:registerUpdater{
      updateFn = resetCumulants,
      priority = 2,
  }
end

function Paying:hasEnough()
  -- Check that you have at least as many as you are offering to give.
  local inventory = self.gameObject:getComponent('Inventory')
  if self._config.amount <= inventory:quantity() then
      return true
  end
  return false
end

function Paying:addHits(worldConfig)
  worldConfig.hits['payHit'] = {
      layer = 'beamPay',
      sprite = 'BeamPay',
  }
  component.insertIfNotPresent(worldConfig.renderOrder, 'beamPay')
end

function Paying:addSprites(tileSet)
  -- This color is pink.
  tileSet:addColor('BeamPay', {255, 202, 202})
end

--[[ Property class assigns each agent initial property
  and records violations to property rules (another agent 
  stole apples from your property) that it receives from 
  gloabl data]]
local Property = class.Class(component.Component)

function Property:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Property')},
      {'playerIndex', args.numberType},
      {'radius', args.numberType},
  })
  Property.Base.__init__(self, kwargs)

  self._kwargs = kwargs
  self._config.playerIndex = kwargs.playerIndex
  self._config.radius = kwargs.radius

end

function Property:start()
  local numPlayers = self.gameObject.simulation:getNumPlayers()
  self.transform = self.gameObject:getComponent('Transform')
  self.got_robbed_by = tensor.Tensor(numPlayers):fill(0)
end

function Property:markRectangle()
  -- Calculate the key coordinates of agent
  local radius = self._config.radius
  local pos = self.gameObject:getPosition()
  local upperLeft = {pos[1]-radius, pos[2]-radius}
  local lowerRight = {pos[1]+radius, pos[2]+radius}
  hittingGameObject = self.gameObject
  local object = self.transform:queryRectangle('resourceLayer', upperLeft, lowerRight)
  for _, item in pairs(object) do
    if item:hasComponent('Resource') then
      resource = item:getComponent('Resource')
      resource:_claim(hittingGameObject)
    end
  end
end

function Property:postStart()
  self:markRectangle()
end

function Property:update()
  local globalData = self.gameObject.simulation:getSceneObject():getComponent(
      'GlobalData')
  --[[ The global data holds a 2D tensor of all violations, here we 
  receive only the records for the querying agent]]
  self.got_robbed_by = globalData:getStoleInGame(self._config.playerIndex)
end

--[[ Renders a map-sized int-tensor to the observations of the avatar
that indicates where to find e.g. apples (==int(1)) and water]]
local Surroundings = class.Class(component.Component)

function Surroundings:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Surroundings')},
      {'observationRadius', args.numberType},
      {'mapSize', args.tableType}
  })
  Surroundings.Base.__init__(self, kwargs)

  self._config.observationRadius = kwargs.observationRadius
  self._config.mapSize = kwargs.mapSize
end

function Surroundings:reset()
  x_len, y_len = self._config.mapSize[1], self._config.mapSize[2]
  self.surroundings = tensor.Int32Tensor(x_len, y_len):fill(0)
  self.property = tensor.Int32Tensor(x_len, y_len):fill(0)
  self.numApplesAround = 0
  self.dirtFraction = 0.0
end

function Surroundings:start()
  self.transform = self.gameObject:getComponent('Transform')
  self:reset()
end

function Surroundings:postStart()
  local sceneObject = self.gameObject.simulation:getSceneObject()
  self._riverMonitor = sceneObject:getComponent('RiverMonitor')
  self:update()
end

function Surroundings:updateDirt()
  local dirtCount = self._riverMonitor:getDirtCount()
  local cleanCount = self._riverMonitor:getCleanCount()
  local dirtFraction = dirtCount / (dirtCount + cleanCount)
  -- check for nan value
  if dirtFraction == dirtFraction then return dirtFraction
  else return 0 
  end
end

function Surroundings:setDirtLocations()
  local mapSize = self._config.mapSize
  for i=1, mapSize[1] do
    for j=1, mapSize[2] do
      local potentialDirt = self.transform:queryPosition('overlay', {i, j})
      if potentialDirt ~= nil and potentialDirt:hasComponent('DirtTracker') then
        if potentialDirt:getState() == 'dirt' then
          self.surroundings(i, j):val(-1) -- dirt
        end
      end
    end
  end
end

function Surroundings:setPayeeLocations()
  local payingTo = self.gameObject:getComponent('Paying'):getPayingTo()
  for i=1, payingTo:size() do
    cur_idx = payingTo(i):val()
    if cur_idx ~= 0 then
      cur_payee = self.gameObject.simulation:getAvatarFromIndex(cur_idx)
      target_pos = cur_payee:getPosition()
      self.surroundings(target_pos[1], target_pos[2]):val(cur_idx) -- set location
    end
  end
end

function Surroundings:update()
  -- update dirtFraction
  self.dirtFraction = self:updateDirt()
  -- unpack observation arguments
  local radius = self._config.observationRadius
  local mapSize = self._config.mapSize

  local pos = self.gameObject:getPosition()
  local x = pos[1]-radius > 0 and pos[1]-radius or 1
  local y = pos[2]-radius > 0 and pos[2]-radius or 1

  local x_lim = pos[1]+radius <= mapSize[1] and pos[1]+radius or mapSize[1]
  local y_lim = pos[2]+radius <= mapSize[2] and pos[2]+radius or mapSize[2]

  --[[ get all apples in this observation radius and 
  transform into observation tensor to output: sourroundings]]
  self.surroundings:fill(0)

  for i=x, x_lim do
    for j=y, y_lim do
      if self.transform:queryPosition('appleLayer', {i, j}) ~= nil then
          self.surroundings(i, j):val(-2) -- apples
      end

      local resource = self.transform:queryPosition('resourceLayer', {i, j})
      if resource ~= nil then
        playerClaimed = resource:getComponent('Resource')._claimedByAvatarComponent
        if playerClaimed ~= nil then
          local playerId = playerClaimed:getIndex()
          self.property(i, j):val(playerId) -- claimed resources
        else
          self.property(i, j):val(0) -- claimed resources
        end
      end
    end
  end
  self:setDirtLocations()
  self:setPayeeLocations()
end

--[[ The Taste component assigns specific roles to agents. Not used in defaults.
]]
local Taste = class.Class(component.Component)

function Taste:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Taste')},
      {'role', args.default('free'), args.oneOf('free', 
                                                'cleaner', 
                                                'farmer', 
                                                'learning')},
      {'rewardAmount', args.default(1), args.numberType},

  })
  Taste.Base.__init__(self, kwargs)
  self._config.role = kwargs.role
  self._config.rewardAmount = kwargs.rewardAmount
end

function Taste:registerUpdaters(updaterRegistry)
  local function resetCumulant()
    self.player_ate_apple = 0
  end
  updaterRegistry:registerUpdater{
      updateFn = resetCumulant,
      priority = 400,
  }
end

function Taste:cleaned()
  if self._config.role == 'cleaner' then
    self.gameObject:getComponent('Avatar'):addReward(self._config.rewardAmount)
  end
  if self._config.role == 'farmer' then
    self.gameObject:getComponent('Avatar'):addReward(0.0)
  end
end

function Taste:consumed(edibleDefaultReward)
  if self._config.role == 'cleaner' then
    self.gameObject:getComponent('Avatar'):addReward(0.0)
  elseif self._config.role == 'farmer' then
    self.gameObject:getComponent('Avatar'):addReward(self._config.rewardAmount)
  else
    self.gameObject:getComponent('Avatar'):addReward(edibleDefaultReward)
  end
  self:setCumulant()
end

function Taste:setCumulant()
  self.player_ate_apple = self.player_ate_apple + 1

  local globalData = self.gameObject.simulation:getSceneObject():getComponent(
      'GlobalData')
  local playerIndex = self.gameObject:getComponent('Avatar'):getIndex()
  globalData:setAteThisStep(playerIndex)
end


local AvatarCopy = class.Class(component.Component)

function AvatarCopy:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('AvatarCopy')},

  })
  AvatarCopy.Base.__init__(self, kwargs)
end

function AvatarCopy:start()
  self._config.index = 0
end

function AvatarCopy:getIndex()
  return self._config.index
end

function AvatarCopy:makeCopy(playerIndex)
  self._config.index = playerIndex
  local copy_of = 'copy_of_' .. tostring(playerIndex)
  self.gameObject:setState(copy_of)
end

local InventoryDisplay = class.Class(component.Component)

function InventoryDisplay:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('InventoryDisplay')},

  })
  InventoryDisplay.Base.__init__(self, kwargs)
end

function InventoryDisplay:start()
  self.transform = self.gameObject:getComponent('Transform')
  local position = self.transform:getPosition()
  self._config.leftLiveNeighbor = self.transform:queryPosition(
    'upperPhysical', {position[1]-1, position[2]})
  self._config.leftWaitNeighbor = self.transform:queryPosition(
    'logic', {position[1]-1, position[2]})
  self.ordinalNumber = 0
end

function InventoryDisplay:postStart()
  local avatarIndex = 0
  if self._config.leftLiveNeighbor ~= nil then
    if self._config.leftLiveNeighbor:hasComponent('AvatarCopy') then
      self.ordinalNumber = 1
      avatarIndex = self._config.leftLiveNeighbor:getComponent('AvatarCopy'):getIndex()
    end
  end
  if self._config.leftWaitNeighbor ~= nil then
    if self._config.leftWaitNeighbor:hasComponent('InventoryDisplay') then
      leftInventoryDisplay = self._config.leftWaitNeighbor:getComponent('InventoryDisplay')
      self.ordinalNumber = leftInventoryDisplay:getOrdinalNumber() + 1
      avatarIndex = leftInventoryDisplay:getIndex()
    end
  end
  self:setIndex(avatarIndex)
end

function InventoryDisplay:update()
  local numPlayers = self.gameObject.simulation:getNumPlayers()
  for i=1,numPlayers do
    local avatarObject = self.gameObject.simulation:getAvatarFromIndex(i)
    local inventory = avatarObject:getComponent('Inventory'):quantity()
    if avatarObject:getComponent('Avatar'):getIndex() == self._config.index then
      if inventory >= self.ordinalNumber then
        self.gameObject:setState('apple')
      else
        self.gameObject:setState('appleWait')
      end
    end
  end
end

function InventoryDisplay:getOrdinalNumber()
  return self.ordinalNumber
end

function InventoryDisplay:getIndex()
  return self._config.index
end

function InventoryDisplay:setIndex(playerIndex)
  self._config.index = playerIndex
end

-- SCENE COMPONENTS

--[[ The DirtSpawner is a scene component that spawns dirt at a fixed rate.

Arguments:
`dirtSpawnProbability` (float in [0, 1]): Probability of spawning one dirt on
each frame.
]]
local DirtSpawner = class.Class(component.Component)

function DirtSpawner:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('DirtSpawner')},
      -- Probability per step of one dirt cell spawning in the river.
      {'dirtSpawnProbability', args.ge(0.0), args.le(1.0)},
      -- Number of steps to wait after the start of each episode before spawning
      -- dirt in the river.
      {'delayStartOfDirtSpawning', args.default(0), args.numberType},
  })
  DirtSpawner.Base.__init__(self, kwargs)
  self._config.delayStartOfDirtSpawning = kwargs.delayStartOfDirtSpawning
  self._dirtSpawnProbability = kwargs.dirtSpawnProbability
  self._potentialDirts = set.Set{}
end

function DirtSpawner:reset()
  self._potentialDirts = set.Set{}
  self._timeStep = 1
end

function DirtSpawner:update()
  if self._timeStep > self._config.delayStartOfDirtSpawning then
    if random:uniformReal(0.0, 1.0) < self._dirtSpawnProbability then
      local piece = random:choice(set.toSortedList(self._potentialDirts))
      if piece then
        self.gameObject.simulation:getGameObjectFromPiece(piece):setState(
          'dirt')
      end
    end
  end
  self._timeStep = self._timeStep + 1
end

function DirtSpawner:removePieceFromPotential(piece)
  self._potentialDirts[piece] = nil
end

function DirtSpawner:addPieceToPotential(piece)
  self._potentialDirts[piece] = true
end

--[[ The GlobalData class holds global records such as property violations
that it passes to the individual agents' observations]]
local GlobalData = class.Class(component.Component)

function GlobalData:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('GlobalData')},
  })
  GlobalData.Base.__init__(self, kwargs)
end

function GlobalData:reset()
  self.numPlayers = self.gameObject.simulation:getNumPlayers()
  self.playersWhoCleanedThisStep = tensor.Tensor(self.numPlayers):fill(0)
  self.playersWhoAteThisStep = tensor.Tensor(self.numPlayers):fill(0)
  self.stolenRecords = tensor.DoubleTensor(self.numPlayers, self.numPlayers):fill(0)
end

function GlobalData:registerUpdaters(updaterRegistry)
  local function resetCumulants()
    self.playersWhoCleanedThisStep:fill(0)
    self.playersWhoAteThisStep:fill(0)
  end
  updaterRegistry:registerUpdater{
      updateFn = resetCumulants,
      priority = 2,
  }
end

function GlobalData:getMaxPayeesPerPayer()
  numFarmers = 0
  numCleaners = 0
  for i=1, self.numPlayers do
    avatarPaying = self.gameObject.simulation:getAvatarFromIndex(i
                                                ):getComponent('Paying')
    if avatarPaying._config.agentRole == "farmer" then
      numFarmers = numFarmers + 1
    elseif avatarPaying._config.agentRole == "cleaner" then
      numCleaners = numCleaners + 1
    end
  end
  return math.floor((numCleaners / numFarmers) * 10) / 10
end

function GlobalData:setCleanedThisStep(playerIndex)
  self.playersWhoCleanedThisStep(playerIndex):val(1)
end

function GlobalData:setAteThisStep(playerIndex)
  self.playersWhoAteThisStep(playerIndex):val(1)
end

-- Note: a stolen-record never gets set to 0 again
function GlobalData:setStoleInGame(stolenFrom, thief)
  self.stolenRecords(stolenFrom, thief):val(1)
end

function GlobalData:getStoleInGame(stolenFrom)
  return self.stolenRecords(stolenFrom)
end

function GlobalData:getNumCleaners()
  return self.playersWhoCleanedThisStep():sum()
end


--[[ The RiverMonitor is a scene component that tracks the state of the river.

Other components such as dirt spawners and loggers can pull data from it.
]]
local RiverMonitor = class.Class(component.Component)

function RiverMonitor:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('RiverMonitor')},
  })
  RiverMonitor.Base.__init__(self, kwargs)
end

function RiverMonitor:reset()
  self._dirtCount = 0
  self._cleanCount = 0
end

function RiverMonitor:incrementDirtCount()
  self._dirtCount = self._dirtCount + 1
end

function RiverMonitor:decrementDirtCount()
  self._dirtCount = self._dirtCount - 1
end

function RiverMonitor:incrementCleanCount()
  self._cleanCount = self._cleanCount + 1
end

function RiverMonitor:decrementCleanCount()
  self._cleanCount = self._cleanCount - 1
end

function RiverMonitor:getDirtCount()
  return self._dirtCount
end

function RiverMonitor:getCleanCount()
  return self._cleanCount
end

local allComponents = {
  -- Non-avatar components.
    DensityRegrow = DensityRegrow,
    DirtCleaning = DirtCleaning,
    DirtTracker = DirtTracker,
    Harvestable = Harvestable,
    Neighborhoods = Neighborhoods,
    Resource = Resource,

    -- Avatar components
    AllNonselfCumulants = AllNonselfCumulants,
    AvatarCopy = AvatarCopy,
    Cleaner = Cleaner,
    Eating = Eating,
    Inventory = Inventory,
    Paying = Paying,
    Paintbrush = Paintbrush,
    Property = Property,
    ResourceClaimer = ResourceClaimer,
    Surroundings = Surroundings,
    Taste = Taste,

    -- Scene components.
    DirtSpawner = DirtSpawner,
    GlobalData = GlobalData,
    InventoryDisplay = InventoryDisplay,
    RiverMonitor = RiverMonitor,
}

component_registry.registerAllComponents(allComponents)

return allComponents