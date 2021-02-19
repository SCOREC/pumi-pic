% MATLAB file for graph generation
clear

%% Read Data
fileID_rebuild = fopen('rebuild_data.dat');
fileID_push = fopen('push_data.dat');

rebuild_data = fscanf(fileID_rebuild, "%d %d %d %d %f", [5 Inf])';
fclose(fileID_rebuild);
push_data = fscanf(fileID_push, "%d %d %d %d %f", [5 Inf])';
fclose(fileID_push);

% element_number, particles_moved, average_time
elms = unique(rebuild_data( rebuild_data(:,1) == 0, 2 ));
%elms = elms ./ 10000; % adjust scale
scs_rebuild = rebuild_data( rebuild_data(:,1) == 0, 3:5 );
csr_rebuild = rebuild_data( rebuild_data(:,1) == 1, 3:5 );
cabm_rebuild = rebuild_data( rebuild_data(:,1) == 2, 3:5 );
scs_push = push_data( push_data(:,1) == 0, 3:5 );
csr_push = push_data( push_data(:,1) == 1, 3:5 );
cabm_push = push_data( push_data(:,1) == 2, 3:5 );

%% Generate Graphs

% {0,1,2,3} = {Evenly,Uniform,Gaussian,Exponential}
% CabM (50%)
cabm_50 = cabm_rebuild( cabm_rebuild(:,2) == 50,:);
cabm_even_times = cabm_50( cabm_50(:,1) == 0, 3);
cabm_uni_times = cabm_50( cabm_50(:,1) == 1, 3);
cabm_gauss_times = cabm_50( cabm_50(:,1) == 2, 3);
cabm_exp_times = cabm_50( cabm_50(:,1) == 3, 3);
% CSR (50%)
csr_50 = csr_rebuild( csr_rebuild(:,2) == 50,:);
csr_even_times = csr_50( csr_50(:,1) == 0, 3);
csr_uni_times = csr_50( csr_50(:,1) == 1, 3);
csr_gauss_times = csr_50( csr_50(:,1) == 2, 3);
csr_exp_times = csr_50( csr_50(:,1) == 3, 3);
% SCS (50%)
scs_50 = scs_rebuild( scs_rebuild(:,2) == 50,:);
scs_even_times = scs_50( scs_50(:,1) == 0, 3);
scs_uni_times = scs_50( scs_50(:,1) == 1, 3);
scs_gauss_times = scs_50( scs_50(:,1) == 2, 3);
scs_exp_times = scs_50( scs_50(:,1) == 3, 3);

% Speedup
figure(1)
hold on
plot( elms, ones(size(elms)) )
plot( elms, scs_even_times./cabm_even_times )
plot( elms, scs_uni_times./cabm_uni_times )
plot( elms, scs_gauss_times./cabm_gauss_times )
plot( elms, scs_exp_times./cabm_exp_times )
xlabel("number elements")
ylabel("SCS/CabM (CabM Speedup)")
legend('One (reference)', 'Even', 'Uniform', 'Gaussian', 'Exponential')
saveas(1,"CabMvSCS_50_rebuild.jpg")

figure(2)
hold on
plot( elms, ones(size(elms)) )
plot( elms, csr_even_times./cabm_even_times )
plot( elms, csr_uni_times./cabm_uni_times )
plot( elms, csr_gauss_times./cabm_gauss_times )
plot( elms, csr_exp_times./cabm_exp_times )
xlabel("number elements")
ylabel("CSR/CabM (CabM Speedup)")
legend('One (reference)', 'Even', 'Uniform', 'Gaussian', 'Exponential')
saveas(2,"CabMvCSR_50_rebuild.jpg")

figure(3)
hold on
hold on
plot( elms, ones(size(elms)) )
plot( elms, scs_even_times./csr_even_times )
plot( elms, scs_uni_times./csr_uni_times )
plot( elms, scs_gauss_times./csr_gauss_times )
plot( elms, scs_exp_times./csr_exp_times )
xlabel("number elements")
ylabel("SCS/CSR (CSR Speedup)")
legend('One (reference)', 'Even', 'Uniform', 'Gaussian', 'Exponential')
saveas(3,"CSRvSCS_50_rebuild.jpg")