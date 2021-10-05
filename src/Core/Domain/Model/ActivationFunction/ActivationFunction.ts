export default abstract class ActivationFunction
{
    protected abstract process(value: number): number;

    public activate(value: number): number
    {
        return this.process(value);
    }
}