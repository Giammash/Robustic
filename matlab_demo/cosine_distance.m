function d = cosine_distance(v1, v2)
    % Compute cosine distance between two vectors
    % d = 1 - cos(theta) = 1 - (v1 . v2) / (|v1| * |v2|)
    %
    % Range: [0, 2]
    %   0 = identical direction
    %   1 = perpendicular (90 degrees)
    %   2 = opposite direction (180 degrees)

    dot_product = dot(v1, v2);
    norm1 = norm(v1);
    norm2 = norm(v2);

    % Add small epsilon to avoid division by zero
    d = 1 - dot_product / (norm1 * norm2 + 1e-8);
end
