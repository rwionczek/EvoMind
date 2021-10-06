import Cell from "./Cell";

export default class Network {
    private inputCells: Cell[];
    private outputCells: Cell[];

    constructor(inputCells: Cell[], outputCells: Cell[]) {
        this.inputCells = inputCells;
        this.outputCells = outputCells;
    }

    public activate(inputSignals: number[]): number[]
    {
        if (inputSignals.length !== this.inputCells.length) {
            throw new Error('Invalid input signals count');
        }

        this.inputCells.forEach((inputCell: Cell, index: number) => {
            inputCell.handleSignal(inputSignals[index]);
        });

        return this.outputCells.map((outputCell: Cell) => {
            return outputCell.getOutputSignalValue();
        });
    }
}
