export default abstract class ActivationFunction
{
    public abstract activate(value: number): number;

    public abstract derivative(value: number): number;
}