function deri = tan_h_output_to_derivative(a)
    % derivative of tanh(x) is 1-tanh(x).^2
    tmp = ones(size(a));
    deri = tmp - (a.^2);
end

