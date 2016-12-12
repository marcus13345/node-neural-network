let colors = require('colors');

//because clamping is a thing that needs to happen
function sigmoid(t) {
    return 1/(1+Math.pow(Math.E, -t));
}

function diomgis(t) {
  return -Math.log(Math.E, ((1/t)-1));
}

// because array transformations are useful
// example
// to create an array of x position from an array of objects with x positions:
// tranform(objectsArray, (v, i, a) => { return v.xPosition; });
// v, i, a = value, index, array.
function transform(array, transformation) {
  var _return = []
  array.forEach((v, i, a) => {
    _return.push(transformation(v, i, a));
  });
  return _return;
}
function transformAssoc(array, transformation) {
  var _return = {}
  for(let name in array) {
    var newObject = transformation(array[name], name, array);
    _return[newObject.key] = newObject.value;
  }
  return _return;
}

//this is the function we will be trying to model
function expect(a, b) {
  return 1 - Math.abs(a - b);
}


// root network class, to store information about the network
// neurons: count of nodes in hidden layer
// diversity: how many of the top mutations are taken to the next generation
// trials: how many mutations are calculated per generation
class BiologicalNeuralNetwork {
  constructor(neurons, diversity, trials) {
    console.log();
    this._inputs = {};
    this._inputSynapses = [];
    this._neurons = [];
    this._outputSynapses = [];
    this._synapses = [];
    this._outputs = {};
    this._generation = 0;
    this._species = 0;
    this._trials = trials;
    this._currentTrial = 0;
    this._diversity = diversity;
    this._currentMutation = [];
    this._generationMutations = [];
    this._savedMutations = [];
    this._locked = false;
    for(let i = 0; i < neurons; i ++) {
      var neuron = new Neuron();
      this._neurons.push(neuron);
    }

    console.log("Created Network with");
    console.log("Neurons:   " + neurons);
    console.log("diversity: " + diversity);
    console.log("trials:    " + trials);
  }

  fitness(value) {
    var currentState = state();
    this._generationMutations.push({
      "state": currentState,
      "fitness": value
    });
  }

  species() {

    var initialState = Math.floor(Math.random() * this._savedMutations.length);
    //set the state to a saved mutation
    setState(this._savedMutations[Math.floor(Math.random() * this._savedMutations.length)]);
    console.log("branching from state: " + initialState);

    //add some randomness as a delta
    var synapseID = Math.floor(Math.random() * this.synapses.length);

    this._synapses[synapseID].value += diomgis(Math.random());
    //return fitness function

    return fitness;
  }

  generation() {
    this._generationMutations.sort((a, b) => {
      return a.fitness > b.fitness ? 1 : a.fitness > b.fitness ? -1 : 0;
    });
    //sort the _generationMutations by fitness
    this._savedMutations = tranform(this._generationMutations.slice(0, 3), (v, i, a) => v.state);
    //take the top 3
    //clear _generationMutations
    this._generationMutations = [];
    //push to saved mutations
  }

  // create the initial savestate.
  // the first mutation.
  // changing input and outputs after here...
  // dont.
  genesis() {
    this._savedMutations.push(this.state());
    this._locked = true;
  }

  state() {
    return tranform(this._synapses, (v, i, a) => v.value);
  }

  setState(state) {
    if(state.length != this._synapses.length) {
      console.warn("given state has " + state.length + " synapses, while the system has " + this._synapses.length);
    }
    for(let i = 0; i < Math.min(state.length, this._synapses.length); i ++) {
      this._synapses[i].value = state[i];
    }
  }

  addInput(name) {
    if(this._locked) {
      console.warn("input: " + name + " not added.\nNetwork is locked!");
    }
    var input = new InputNode();
    input.setName(name);
    this._inputs[name] = input;
    for(let i = 0; i < this._neurons.length; i ++) {
      var synapse = new Synapse(input);

      this._inputSynapses.push(synapse);
      this._synapses.push(synapse);
      this._neurons[i].addSynapse(synapse);
    }
  }

  addOutput(name) {
    if(this._locked) {
      console.warn("input: " + name + " not added.\nNetwork is locked!");
    }
    var output = new OutputNeuron();
    output.setName(name);
    this._outputs[name] = output;
    for(let i = 0; i < this._neurons.length; i ++) {
      var synapse = new Synapse(this._neurons[i]);
      this._synapses.push(synapse);
      this._outputSynapses.push(synapse);
      output.addSynapse(synapse);
    }
  }

  predict(inputs) {
    //DO THE NETWORK

    //input the input information
    for(let name in inputs) {
      this._inputs[name].value = inputs[name];
    }

    //calculate the input synapses
    this._inputSynapses.forEach((v, i, a) => {
      v.calculate();
    });

    //calculate the inner layer
    this._neurons.forEach((v, i, a) => {
      v.calculate();
    });


    //calculate the output synapses
    this._outputSynapses.forEach((v, i, a) => {
      v.calculate();
    });

    var best = null;
    for(let name in this._outputs) {
      var node = this._outputs[name];
      node.calculate();
      if(best == null || node.confidence > best.confidence)
        best = node;
    }
    var outs = transformAssoc(this._outputs, (v, name) => {return {"key": name, "value": v.confidence}; });
    // console.log(outs, this._outputs, outs.length);
    best.notifyBest();
    return outs;
  }
}


//represents all nodes: input, output, and synapse
class Node {
  constructor() {
    this._confidence = 0;
    this._name = "Synapse";
  }

  get confidence() {
    return this._confidence;
  }

  set confidence(val) {
    this._confidence = val;
  }

  setName(val) {
    this._name = val;
  }

  resetConfindence() {
    this._confidence = 0;
  }
}

class InputNode extends Node{
  constructor() {
    super();
  }
}

class Neuron extends Node{
  constructor() {
    super();
    this._synapses = [];
  }

  addSynapse(val) {
    this._synapses.push(val);
  }

  calculate() {
    var sum = 0;
    this._synapses.forEach((v, i, a) => {
      sum += v.value;
    });
    this.confidence = sigmoid(sum);
    return this.confidence;
  }
}

class Synapse {
  constructor(input) {
    this._input = input;
    this._weight = 3 * (Math.random() - 0.5);
    this._value = 0;

    // console.log();
    // console.log("Synapse created");
    // console.log("Weight: " + this._weight.toString().red);
    // console.log("Input:  " + this._input._name.toString().red);
  }

  addInput(input) {
    this._inputs.push(input);
  }

  get value() {
    return this._value;
  }

  set value(val) {
    this._value = val;
  }

  calculate() {
    this._value = this._input.confidence * this._weight
  }
}



class OutputNeuron extends Neuron{
  constructor() {
    super();
    this.bestCallback = null;
  }

  on(event, callback) {
    if(event == 'best') {
      this.bestCallback = callback;
    }
  }

  notifyBest() {
    if(this.bestCallback) {
      this.bestCallback();
    }
  }
}


// var clear = console.clear || function(){process.stdout.write('\033c');};
// clear();
// console.log("Node Neural Networking Started");
//
// var network = new BiologicalNeuralNetwork(10, 5, 20)
// network.addInput("a");
// network.addInput("b");
// network.addOutput("result");
//
// console.log();
// console.log("Input Synapses:  " + network._inputSynapses.length);
// console.log("Output Synapses: " + network._outputSynapses.length);
//
// for(let i = 0; i < 10; i ++) {
//   console.log(network.predict({"a": 1, "b": 0}));
// }
