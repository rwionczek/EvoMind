import ActivationFunction from "./ActivationFunction";

export default class ReluActivationFunction extends ActivationFunction
{
    activate(value: number): number {
        return value < 0.0 ? 0.0 : value;
    }

    derivative(value: number): number {
        return value < 0.0 ? 0.0 : 1.0;
    }
}
