import Cell from "./Cell";

export default class Connection {
    private readonly weight: number;
    private inputCell: Cell;
    private outputCell: Cell;

    constructor(inputCell: Cell, outputCell: Cell, weight: number) {
        this.inputCell = inputCell;
        this.outputCell = outputCell;
        this.weight = weight;
    }

    public sendSignal(value: number): void
    {
        this.outputCell.handleSignal(value * this.weight);
    }
}
