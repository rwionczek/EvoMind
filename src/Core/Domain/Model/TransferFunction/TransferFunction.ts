export default abstract class TransferFunction
{
    protected abstract process(cellValue: number, signalValue: number): number;

    public transfer(cellValue: number, signalValue: number): number
    {
        return this.process(cellValue, signalValue);
    }
}