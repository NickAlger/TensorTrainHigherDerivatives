nn = [41,42,43,44,45];
% nn = [41,42,43];
d = length(nn);

T = make_hilbert_tensor(nn);
get_T_entry = @(ind) get_entry(T, ind);

tt_cross_extra_rank_constant = 2;
rr0 = 1:19;
rr = zeros(size(rr0));
errs = zeros(size(rr0));
for k=1:length(rr0)
    r0 = rr0(k);
    CC_cross = dmrg_cross(d, nn, get_T_entry, 1e-15, 'verb',false, 'maxr', r0);
    r = max(rank(CC_cross));
    rr(k) = r;
    T_cross = reshape(full(CC_cross), nn);
    err = norm(T_cross(:) - T(:))/norm(T(:));
    errs(k) = err;
    disp(['r=', num2str(r), ', err=', num2str(err)])
end

% save('hilbert_dmrg_cross_data0.mat', 'nn', 'tols', 'rr', 'errs')
% save('hilbert_dmrg_cross_data1.mat', 'nn', 'tols', 'rr', 'errs')
save('hilbert_dmrg_cross_data2.mat', 'nn', 'rr', 'errs')
