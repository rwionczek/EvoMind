import ActivationFunction from "./ActivationFunction";

export default class SigmoidActivationFunction extends ActivationFunction
{
    activate(value: number): number {
        return 1.0 / (1 + Math.exp(-value));
    }

    derivative(value: number): number {
        return value * (1.0 - value);
    }
}
