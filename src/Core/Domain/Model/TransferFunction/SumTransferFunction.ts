import TransferFunction from "./TransferFunction";

export default class SumTransferFunction extends TransferFunction
{
    process(cellValue: number, signalValue: number): number {
        return cellValue + signalValue;
    }
}
