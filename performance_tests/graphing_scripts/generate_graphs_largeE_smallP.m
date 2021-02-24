% MATLAB file for graph generation
clear

%% Data Reading
fileID_rebuild = fopen('data/largeE_smallP_rebuild.dat');
fileID_push = fopen('data/largeE_smallP_push.dat');

rebuild_data = fscanf(fileID_rebuild, "%d %d %d %d %f", [5 Inf])';
fclose(fileID_rebuild);
push_data = fscanf(fileID_push, "%d %d %d %d %f", [5 Inf])';
fclose(fileID_push);

% element_number, particles_moved, average_time
elms = unique(rebuild_data( rebuild_data(:,1) == 0, 2 ));
scs_rebuild = rebuild_data( rebuild_data(:,1) == 0, 3:5 );
csr_rebuild = rebuild_data( rebuild_data(:,1) == 1, 3:5 );
cabm_rebuild = rebuild_data( rebuild_data(:,1) == 2, 3:5 );
scs_push = push_data( push_data(:,1) == 0, 3:5 );
csr_push = push_data( push_data(:,1) == 1, 3:5 );
cabm_push = push_data( push_data(:,1) == 2, 3:5 );

%% Data Filtering
% {0,1,2,3} = {Evenly,Uniform,Gaussian,Exponential}

% CabM (50%)
cabm_50 = cabm_rebuild( cabm_rebuild(:,2) == 50,:);
cabm_even_50 = cabm_50( cabm_50(:,1) == 0, 3);
cabm_uni_50 = cabm_50( cabm_50(:,1) == 1, 3);
cabm_gauss_50 = cabm_50( cabm_50(:,1) == 2, 3);
cabm_exp_50 = cabm_50( cabm_50(:,1) == 3, 3);
% CSR (50%)
csr_50 = csr_rebuild( csr_rebuild(:,2) == 50,:);
csr_even_50 = csr_50( csr_50(:,1) == 0, 3);
csr_uni_50 = csr_50( csr_50(:,1) == 1, 3);
csr_gauss_50 = csr_50( csr_50(:,1) == 2, 3);
csr_exp_50 = csr_50( csr_50(:,1) == 3, 3);
% SCS (50%)
scs_50 = scs_rebuild( scs_rebuild(:,2) == 50,:);
scs_even_50 = scs_50( scs_50(:,1) == 0, 3);
scs_uni_50 = scs_50( scs_50(:,1) == 1, 3);
scs_gauss_50 = scs_50( scs_50(:,1) == 2, 3);
scs_exp_50 = scs_50( scs_50(:,1) == 3, 3);

% CabM (Pseudo-Push)
cabm_push = cabm_push( cabm_push(:,2) == 50,:);
cabm_even_push = cabm_push( cabm_push(:,1) == 0, 3);
cabm_uni_push = cabm_push( cabm_push(:,1) == 1, 3);
cabm_gauss_push = cabm_push( cabm_push(:,1) == 2, 3);
cabm_exp_push = cabm_push( cabm_push(:,1) == 3, 3);
% CSR (Pseudo-Push)
csr_push = csr_push( csr_push(:,2) == 50,:);
csr_even_push = csr_push( csr_push(:,1) == 0, 3);
csr_uni_push = csr_push( csr_push(:,1) == 1, 3);
csr_gauss_push = csr_push( csr_push(:,1) == 2, 3);
csr_exp_push = csr_push( csr_push(:,1) == 3, 3);
% SCS (Pseudo-Push)
scs_push = scs_push( scs_push(:,2) == 50,:);
scs_even_push = scs_push( scs_push(:,1) == 0, 3);
scs_uni_push = scs_push( scs_push(:,1) == 1, 3);
scs_gauss_push = scs_push( scs_push(:,1) == 2, 3);
scs_exp_push = scs_push( scs_push(:,1) == 3, 3);

%% Graph Generation
% Even
% figure(1)
% semilogy( ...
%     elms, scs_even_50./cabm_even_50, 'r--', ... % CabM Rebuild 50%
%     elms, scs_even_push./cabm_even_push, 'r:', ... % CabM Pseudo-Push
%     elms, scs_even_50./csr_even_50, 'b--', ... % CSR Rebuild 50%
%     elms, scs_even_push./csr_even_push, 'b:', ... % CSR Pseudo-Push
%     elms, ones(size(elms)), 'k', 'LineWidth', 0.75 ) % Reference
% ax = gca;
% ax.XAxis.Exponent = 0;
% ax.XTick = 0:2500:20000;
% ax.YGrid = 'on';
% xlabel( {'Number Particles (Thousands)','Number Elements'} )
% ylabel("Structure Speedup (SCS/Structure)")
% legend('CabM Rebuild 50%', 'CabM Pseudo-Push', ...
%      'CSR Rebuild 50%', 'CSR Pseudo-Push', ...
%     'SCS (Reference)', 'Location', 'southeast')
% title({'Speedup (Even Distribution)','1:1,000 Element to Particle Ratio'})
% saveas(1,"largeE_smallP_even.png")

% Uniform
figure(2)
semilogy( ...
	elms, scs_uni_50./cabm_uni_50, 'r--', ... % CabM Rebuild 50%
	elms, scs_uni_push./cabm_uni_push, 'r:', ... % CabM Pseudo-Push
	elms, scs_uni_50./csr_uni_50, 'b--', ... % CSR Rebuild 50%
	elms, scs_uni_push./csr_uni_push, 'b:', ... % CSR Pseudo-Push
	elms, ones(size(elms)), 'k', 'LineWidth', 0.75 ) % Reference
ax = gca;
ax.XAxis.Exponent = 0;
ax.XTick = 0:2500:20000;
ax.YGrid = 'on';
xlabel( {'Number Particles (Thousands)','Number Elements'} )
ylabel("Structure Speedup (SCS/Structure)")
legend('CabM Rebuild 50%', 'CabM Pseudo-Push', ...
     'CSR Rebuild 50%', 'CSR Pseudo-Push', ...
    'SCS (Reference)', 'Location', 'southeast')
title({'Speedup (Uniform Distribution)','1:1,000 Element to Particle Ratio'})
saveas(2,"largeE_smallP_uniform.png")

% Gaussian
figure(3)
semilogy( ...
	elms, scs_gauss_50./cabm_gauss_50, 'r--', ... % CabM Rebuild 50%
	elms, scs_gauss_push./cabm_gauss_push, 'r:', ... % CabM Pseudo-Push
	elms, scs_gauss_50./csr_gauss_50, 'b--', ... % CSR Rebuild 50%
	elms, scs_gauss_push./csr_gauss_push, 'b:', ... % CSR Pseudo-Push
	elms, ones(size(elms)), 'k', 'LineWidth', 0.75 ) % Reference
ax = gca;
ax.XAxis.Exponent = 0;
ax.XTick = 0:2500:20000;
ax.YGrid = 'on';
xlabel( {'Number Particles (Thousands)','Number Elements'} )
ylabel("Structure Speedup (SCS/Structure)")
legend('CabM Rebuild 50%', 'CabM Pseudo-Push', ...
     'CSR Rebuild 50%', 'CSR Pseudo-Push', ...
    'SCS (Reference)', 'Location', 'southeast')
title({'Speedup (Gaussian Distribution)','1:1,000 Element to Particle Ratio'})
saveas(3,"largeE_smallP_gaussian.png")

% Exponential
figure(4)
semilogy( ...
    elms, scs_exp_50./cabm_exp_50, 'r--', ... % CabM Rebuild 50%
    elms, scs_exp_push./cabm_exp_push, 'r:', ... % CabM Pseudo-Push
	elms, scs_exp_50./csr_exp_50, 'b--', ... % CSR Rebuild 50%
	elms, scs_exp_push./csr_exp_push, 'b:', ... % CSR Pseudo-Push
	elms, ones(size(elms)), 'k', 'LineWidth', 0.75 )
ax = gca;
ax.XAxis.Exponent = 0;
ax.XTick = 0:2500:20000;
ax.YGrid = 'on';
xlabel( {'Number Particles (Thousands)','Number Elements'} )
ylabel("Structure Speedup (SCS/Structure)")
legend('CabM Rebuild 50%', 'CabM Pseudo-Push', ...
     'CSR Rebuild 50%', 'CSR Pseudo-Push', ...
    'SCS (Reference)', 'Location', 'southeast')
title({'Speedup (Exponential Distribution)','1:1,000 Element to Particle Ratio'})
saveas(4,"largeE_smallP_exponential.png")