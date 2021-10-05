import {expect} from "chai";
import Cell from "../../../../src/Core/Domain/Model/Cell";
import SumTransferFunction from "../../../../src/Core/Domain/Model/TransferFunction/SumTransferFunction";
import ReluActivationFunction from "../../../../src/Core/Domain/Model/ActivationFunction/ReluActivationFunction";

describe('Cell', function() {
  it('should handle signal', function() {
    const outputCell = new Cell(new SumTransferFunction(), new ReluActivationFunction());

    const inputCell = new Cell(new SumTransferFunction(), new ReluActivationFunction());
    inputCell.connectOutputCell(outputCell, 5.0);

    inputCell.handleSignal(2.0);

    expect(outputCell.getOutputSignalValue()).to.equal(10.0);
  });

  it('should include bias in output signal', function() {
    const outputCell = new Cell(new SumTransferFunction(), new ReluActivationFunction());
    outputCell.setBias(3.0);

    const inputCell = new Cell(new SumTransferFunction(), new ReluActivationFunction());
    inputCell.connectOutputCell(outputCell, 5.0);

    inputCell.handleSignal(2.0);

    expect(outputCell.getOutputSignalValue()).to.equal(13.0);
  });

  it('should propagate signal to connected output cells', function() {
    const outputCell = new Cell(new SumTransferFunction(), new ReluActivationFunction());

    const middleCell = new Cell(new SumTransferFunction(), new ReluActivationFunction());
    middleCell.connectOutputCell(outputCell, 5.0);

    const inputCell = new Cell(new SumTransferFunction(), new ReluActivationFunction());
    inputCell.connectOutputCell(middleCell, 3.0);

    inputCell.handleSignal(2.0);

    expect(outputCell.getOutputSignalValue()).to.equal(30.0);
  });

  it('should propagate signal to multiple connected output cells', function() {
    const firstOutputNeuron = new Cell(new SumTransferFunction(), new ReluActivationFunction());
    const secondOutputNeuron = new Cell(new SumTransferFunction(), new ReluActivationFunction());

    const inputCell = new Cell(new SumTransferFunction(), new ReluActivationFunction());
    inputCell.connectOutputCell(firstOutputNeuron, 3.0);
    inputCell.connectOutputCell(secondOutputNeuron, 5.0);

    inputCell.handleSignal(2.0);

    expect(firstOutputNeuron.getOutputSignalValue()).to.equal(6.0);
    expect(secondOutputNeuron.getOutputSignalValue()).to.equal(10.0);
  });

  it('should sum signal from multiple connected input cells', function() {
    const outputCell = new Cell(new SumTransferFunction(), new ReluActivationFunction());

    const firstInputNeuron = new Cell(new SumTransferFunction(), new ReluActivationFunction());
    firstInputNeuron.connectOutputCell(outputCell, 3.0);

    const secondInputNeuron = new Cell(new SumTransferFunction(), new ReluActivationFunction());
    secondInputNeuron.connectOutputCell(outputCell, 5.0);

    firstInputNeuron.handleSignal(2.0);
    secondInputNeuron.handleSignal(4.0);

    expect(outputCell.getOutputSignalValue()).to.equal(26.0);
  });

  it('should output sum signal from multiple connected input cells', function() {
    const middleCell = new Cell(new SumTransferFunction(), new ReluActivationFunction());

    const firstInputNeuron = new Cell(new SumTransferFunction(), new ReluActivationFunction());
    firstInputNeuron.connectOutputCell(middleCell, 3.0);

    const secondInputNeuron = new Cell(new SumTransferFunction(), new ReluActivationFunction());
    secondInputNeuron.connectOutputCell(middleCell, 5.0);

    const outputCell = new Cell(new SumTransferFunction(), new ReluActivationFunction());

    middleCell.connectOutputCell(outputCell, 2.0);

    firstInputNeuron.handleSignal(2.0);
    secondInputNeuron.handleSignal(4.0);

    expect(outputCell.getOutputSignalValue()).to.equal(52.0);
  });
});
