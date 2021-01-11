import { expect } from "chai";
import Brain from "../src/Brain";
import Input from "../src/Input";

describe('Primitive brain behavior', function() {
  it('should do nothing for first cycle', function() {
    const brain = new Brain(1, 1);

    const outputs = brain.execute([new Input(0.5)]);

    expect(outputs.map(output => output.value)).to.deep.equal([0.0]);
  });

  it('should do nothing for multiple cycles when inputs are weak ', function() {
    const brain = new Brain(1, 1);

    brain.execute([new Input(0.1)]);
    brain.execute([new Input(0.15)]);
    const outputs = brain.execute([new Input(0.05)]);

    expect(outputs.map(output => output.value)).to.deep.equal([0.0]);
  });

  it('should do something when inputs are strong', function() {
    const brain = new Brain(1, 1);

    brain.execute([new Input(0.6)]);
    brain.execute([new Input(0.8)]);
    const outputs = brain.execute([new Input(0.7)]);

    expect(outputs[0].value).to.be.above(0.0);
  });
});
