export default abstract class TransferFunction
{
    public abstract transfer(cellValue: number, signalValue: number): number;
}