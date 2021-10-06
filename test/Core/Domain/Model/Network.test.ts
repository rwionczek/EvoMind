import {expect} from "chai";
import Cell from "../../../../src/Core/Domain/Model/Cell";
import SumTransferFunction from "../../../../src/Core/Domain/Model/TransferFunction/SumTransferFunction";
import ReluActivationFunction from "../../../../src/Core/Domain/Model/ActivationFunction/ReluActivationFunction";
import Network from "../../../../src/Core/Domain/Model/Network";

describe('Cell', function() {
  it('should propagate signal to output layer', function() {
    const firstInputLayerCell = new Cell(new SumTransferFunction(), new ReluActivationFunction());
    const secondInputLayerCell = new Cell(new SumTransferFunction(), new ReluActivationFunction());

    const middleLayerCell = new Cell(new SumTransferFunction(), new ReluActivationFunction());
    middleLayerCell.setBias(5.0);

    const firstOutputLayerCell = new Cell(new SumTransferFunction(), new ReluActivationFunction());
    const secondOutputLayerCell = new Cell(new SumTransferFunction(), new ReluActivationFunction());

    firstInputLayerCell.connectOutputCell(middleLayerCell, 2.0);
    secondInputLayerCell.connectOutputCell(middleLayerCell, 3.0);

    middleLayerCell.connectOutputCell(firstOutputLayerCell, 2.0);
    middleLayerCell.connectOutputCell(secondOutputLayerCell, 1.0);

    const network = new Network(
        [firstInputLayerCell, secondInputLayerCell],
        [firstOutputLayerCell, secondOutputLayerCell],
    );

    expect(network.activate([2.0, 3.0])).to.eql([36.0, 18.0]);
  });
});
