import TransferFunction from "./TransferFunction";

export default class SumTransferFunction extends TransferFunction
{
    transfer(cellValue: number, signalValue: number): number {
        return cellValue + signalValue;
    }
}
