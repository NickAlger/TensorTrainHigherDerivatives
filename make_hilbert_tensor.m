function T = make_hilbert_tensor(nn)

d = length(nn);
T = zeros(nn);
cc = cell(1,d);
[cc{:}] = ind2sub(size(T), (1:numel(T))');
T(:) = 1./sum(cell2mat(cc),2);