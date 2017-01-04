function out = prox_dual(in, step, prox_primal)
    out = in - step*prox_primal((1/step)*in, 1/step);
end

