function [alignment_cost, cost_matrix, path] = dtw_cosine(template1, template2)
    % Dynamic Time Warping with Cosine Distance
    %
    % Inputs:
    %   template1: N x nch matrix (N time windows, nch channels/features)
    %   template2: M x nch matrix (M time windows, nch channels/features)
    %
    % Outputs:
    %   alignment_cost: scalar DTW distance (LOWER = MORE SIMILAR)
    %   cost_matrix: accumulated cost matrix (for visualization)
    %   path: optimal warping path

    [N, ~] = size(template1);
    [M, ~] = size(template2);

    % Initialize cost matrix with infinity
    % Size is (N+1) x (M+1)
    cost_matrix = inf(N + 1, M + 1);
    cost_matrix(1, 1) = 0;

    % Traceback matrix for path reconstruction (optional, for visualization)
    traceback = zeros(N, M);

    % MAIN DTW LOOP
  
    for i = 1:N
        for j = 1:M
            
            % Step 1: Compute local cost (cosine distance between windows)
            % template1(i, :) = feature vector at window i of template 1
            % template2(j, :) = feature vector at window j of template 2
            dist = cosine_distance(template1(i,:), template2(j,:));

            % -----------------------------------------------------------------
            % Step 2: time warping
            % cost(i,j) = local_cost + min(cost from three neighbors)
            % -----------------------------------------------------------------
            % Three possible predecessors:
            %   - Diagonal (i-1, j-1): "match" - both templates advance
            %   - Vertical (i-1, j):   "insertion" - template1 advances
            %   - Horizontal (i, j-1): "deletion" - template2 advances
            candidates = [
                cost_matrix(i, j),     ... % diagonal
                cost_matrix(i, j + 1),  ... % vertical
                cost_matrix(i + 1, j)   % horizontal
            ];

            [min_cost, direction] = min(candidates);
            cost_matrix(i + 1, j + 1) = dist + min_cost;
            traceback(i, j) = direction;
        end
    end

    % alignment cost is the total accumulated cost at (N, M)
    alignment_cost = cost_matrix(N + 1, M + 1);

    % Reconstruct path for visualization
    path = reconstruct_path(traceback, N, M);
end


