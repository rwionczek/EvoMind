import { expect } from "chai";
import Brain from "../src/Brain";
import Input from "../src/Input";

describe('Brain short term memory', function() {
  it('should remember inputs and outputs', function() {
    const brain = new Brain(2, 1);

    brain.execute([new Input(0.3), new Input(0.5)]);
    brain.execute([new Input(0.6), new Input(0.2)]);
    brain.execute([new Input(0.7), new Input(0.1)]);

    const shortTermMemorySnapshots = brain.getShortTermMemorySnapshots();

    expect(shortTermMemorySnapshots).to.have.length(3);
    expect(shortTermMemorySnapshots[0].inputs.map(input => input.value)).to.deep.equal([0.7, 0.1]);
    expect(shortTermMemorySnapshots[1].inputs.map(input => input.value)).to.deep.equal([0.6, 0.2]);
    expect(shortTermMemorySnapshots[2].inputs.map(input => input.value)).to.deep.equal([0.3, 0.5]);
  });
});
