import Connection from "./Connection";
import ActivationFunction from "./ActivationFunction/ActivationFunction";
import TransferFunction from "./TransferFunction/TransferFunction";

export default class Cell {
    private bias: number = 0.0;
    private activationFunction: ActivationFunction;
    private transferFunction: TransferFunction;

    private currentValue: number = 0.0;
    private outputSignalValue: number = 0.0;
    private receivedSignals: number = 0;

    private inputConnections: Connection[] = [];
    private outputConnections: Connection[] = [];

    constructor(transferFunction: TransferFunction, activationFunction: ActivationFunction) {
        this.activationFunction = activationFunction;
        this.transferFunction = transferFunction;
    }

    public handleSignal(signalValue: number): void
    {
        this.currentValue = this.transferFunction.transfer(this.currentValue, signalValue);

        ++this.receivedSignals;

        if (this.isAllSignalReceived()) {
            this.outputSignalValue = this.activationFunction.activate(this.currentValue) + this.bias;

            this.outputConnections.forEach((inputConnection) => {
                inputConnection.sendSignal(this.outputSignalValue);
            });
        }
    }

    public setBias(bias: number): void
    {
        this.bias = bias;
    }

    public getOutputSignalValue(): number
    {
        return this.outputSignalValue;
    }

    public addInputConnection(connection: Connection): void
    {
        this.inputConnections.push(connection);
    }

    public addOutputConnection(connection: Connection): void
    {
        this.outputConnections.push(connection);
    }

    public connectOutputCell(cell: Cell, weight: number): void
    {
        const connection = new Connection(this, cell, weight);

        this.addOutputConnection(connection);
        cell.addInputConnection(connection);
    }

    private isAllSignalReceived(): boolean
    {
        return this.receivedSignals >= this.inputConnections.length;
    }
}
