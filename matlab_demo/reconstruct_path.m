function path = reconstruct_path(traceback, N, M)
    % RECONSTRUCT_PATH Trace back through the cost matrix
    %
    % Returns the optimal alignment path from (1,1) to (N,M)

    path = [N, M];
    i = N;
    j = M;

    while i > 1 || j > 1
        if i == 1
            j = j - 1;
        elseif j == 1
            i = i - 1;
        else
            direction = traceback(i, j);
            if direction == 1      % came from diagonal
                i = i - 1;
                j = j - 1;
            elseif direction == 2  % came from above (vertical)
                i = i - 1;
            else                   % came from left (horizontal)
                j = j - 1;
            end
        end
        path = [i, j; path];
    end
end
