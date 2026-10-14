function l_count = countMNISTlabels(labels)
% count the 0 throuh 9s
    l_count = accumarray(labels + 1, ones(size(labels)), [10,1]);
end