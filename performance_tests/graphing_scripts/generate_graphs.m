function err = generate_graphs(name)
% GENERATE_GRAPHS  plot the timings of push, migrate and rebuild
%                  for each particle structure and distribution
%   err = GENERATE_GRAPHS(name) plots the results for the given
%                               input file to <name>_plot.png
%

err=1;

rebuildInputFile = strcat(name,'_rebuild.dat');
pushInputFile = strcat(name,'_push.dat');
migrateInputFile = strcat(name,'_migrate.dat');
inputFiles = {rebuildInputFile pushInputFile migrateInputFile};

for filename = inputFiles
    if not(isfile(filename{1}))
        error('file %s does not exist', filename{1})
    end
end

% Excluded: CSR Migrate, Even Distribution

YTick = [0.01,0.1,0.5,1,5,10,100,1000];
YTickLabel = {'0.01','0.1x','0.5x','1x','5x','10x','100x', '1000x'};
LineWidth = 2;

outname = strcat(name,'plots.png');

%% Data Reading
nFiles = size(inputFiles,2);
inputFileIds = cell(1, nFiles);
for i = 1:nFiles
  inputFileIds{i} = fopen(inputFiles{i});
end

% todo read these from the header
% functions - this is selecting array indices
REBUILD=1;
PUSH=2;
MIGRATE=3;
NUMFUNC=3;
% structures - selecting array values
SCS=0;
CSR=1;
CABM=2;
DPS=3;
NUMSTRUCT=4;
% distributions - selecting array values
EVEN=0;
UNIFORM=1;
GAUSS=2;
EXPONENTIAL=3;
NUMDIST=4;

% remove header
for fileId = inputFileIds
    for i = 1:3
        fgetl(fileId{1});
    end
end

% struct, element_number, distribution, average_time
rebuild_data = readFileDataAndClose(inputFileIds{1});
push_data = readFileDataAndClose(inputFileIds{2});
migrate_data = readFileDataAndClose(inputFileIds{3});


%% Data Filtering

% find length of graphs
structLen = getStructureLengths(rebuild_data);
elms = getElms(rebuild_data, structLen);

allData = {rebuild_data push_data migrate_data};
mat = getMatrix(allData,NUMFUNC,NUMSTRUCT,NUMDIST,length(elms));

% pull distributions and times
scs_rebuild = getFunctionTimeForStructure(rebuild_data,SCS);
csr_rebuild = getFunctionTimeForStructure(rebuild_data,CSR);
cabm_rebuild = getFunctionTimeForStructure(rebuild_data,CABM);
dps_rebuild = getFunctionTimeForStructure(rebuild_data,DPS);
scs_push = getFunctionTimeForStructure(push_data,SCS);
csr_push = getFunctionTimeForStructure(push_data,CSR);
cabm_push = getFunctionTimeForStructure(push_data,CABM);
dps_push = getFunctionTimeForStructure(push_data,DPS);
scs_migrate = getFunctionTimeForStructure(migrate_data,SCS);
csr_migrate = getFunctionTimeForStructure(migrate_data,CSR);
cabm_migrate = getFunctionTimeForStructure(migrate_data,CABM);
dps_migrate = getFunctionTimeForStructure(migrate_data,DPS);

% Separate data by distribution, {0,1,2,3} = {Evenly,Uniform,Gaussian,Exponential}

% SCS Rebuild
%scs_rebuild_even = scs_rebuild( scs_rebuild(:,1) == 0, 2);

scs_rebuild_uni = getTime(allData, REBUILD, SCS, UNIFORM)
scs_rebuild_gauss = getTime(rebuild_data, SCS, GAUSS);
scs_rebuild_exp = getTime(rebuild_data, SCS, EXPONENTIAL);
% CSR Rebuild
%csr_rebuild_even = csr_rebuild( csr_rebuild(:,1) == 0, 2);
csr_rebuild_uni = csr_rebuild( csr_rebuild(:,1) == 1, 2);
csr_rebuild_gauss = csr_rebuild( csr_rebuild(:,1) == 2, 2);
csr_rebuild_exp = csr_rebuild( csr_rebuild(:,1) == 3, 2);
% CabM Rebuild
%cabm_rebuild_even = cabm_rebuild( cabm_rebuild(:,1) == 0, 2);
cabm_rebuild_uni = cabm_rebuild( cabm_rebuild(:,1) == 1, 2);
cabm_rebuild_gauss = cabm_rebuild( cabm_rebuild(:,1) == 2, 2);
cabm_rebuild_exp = cabm_rebuild( cabm_rebuild(:,1) == 3, 2);
% DPS Rebuild
%dps_rebuild_even = dps_rebuild( dps_rebuild(:,1) == 0, 2);
dps_rebuild_uni = dps_rebuild( dps_rebuild(:,1) == 1, 2);
dps_rebuild_gauss = dps_rebuild( dps_rebuild(:,1) == 2, 2);
dps_rebuild_exp = dps_rebuild( dps_rebuild(:,1) == 3, 2);

% SCS Pseudo-Push
%scs_push_even = scs_push( scs_push(:,1) == 0, 2);
scs_push_uni = scs_push( scs_push(:,1) == 1, 2);
scs_push_gauss = scs_push( scs_push(:,1) == 2, 2);
scs_push_exp = scs_push( scs_push(:,1) == 3, 2);
% CSR Pseudo-Push
%csr_push_even = csr_push( csr_push(:,1) == 0, 2);
csr_push_uni = csr_push( csr_push(:,1) == 1, 2);
csr_push_gauss = csr_push( csr_push(:,1) == 2, 2);
csr_push_exp = csr_push( csr_push(:,1) == 3, 2);
% CabM Pseudo-Push
%cabm_push_even = cabm_push( cabm_push(:,1) == 0, 2);
cabm_push_uni = cabm_push( cabm_push(:,1) == 1, 2);
cabm_push_gauss = cabm_push( cabm_push(:,1) == 2, 2);
cabm_push_exp = cabm_push( cabm_push(:,1) == 3, 2);
% DPS Pseudo-Push
%dps_push_even = dps_push( dps_push(:,1) == 0, 2);
dps_push_uni = dps_push( dps_push(:,1) == 1, 2);
dps_push_gauss = dps_push( dps_push(:,1) == 2, 2);
dps_push_exp = dps_push( dps_push(:,1) == 3, 2);

% SCS Migrate
%scs_migrate_even = scs_migrate( scs_migrate(:,1) == 0, 2);
scs_migrate_uni = scs_migrate( scs_migrate(:,1) == 1, 2);
scs_migrate_gauss = scs_migrate( scs_migrate(:,1) == 2, 2);
scs_migrate_exp = scs_migrate( scs_migrate(:,1) == 3, 2);
% % CSR Migrate
%csr_migrate_even = csr_migrate( csr_migrate(:,1) == 0, 2);
csr_migrate_uni = csr_migrate( csr_migrate(:,1) == 1, 2);
csr_migrate_gauss = csr_migrate( csr_migrate(:,1) == 2, 2);
csr_migrate_exp = csr_migrate( csr_migrate(:,1) == 3, 2);
% CabM Migrate
%cabm_migrate_even = cabm_migrate( cabm_migrate(:,1) == 0, 2);
cabm_migrate_uni = cabm_migrate( cabm_migrate(:,1) == 1, 2);
cabm_migrate_gauss = cabm_migrate( cabm_migrate(:,1) == 2, 2);
cabm_migrate_exp = cabm_migrate( cabm_migrate(:,1) == 3, 2);
% DPS Migrate
%dps_migrate_even = dps_migrate( dps_migrate(:,1) == 0, 2);
dps_migrate_uni = dps_migrate( dps_migrate(:,1) == 1, 2);
dps_migrate_gauss = dps_migrate( dps_migrate(:,1) == 2, 2);
dps_migrate_exp = dps_migrate( dps_migrate(:,1) == 3, 2);

%% Graph Generation

% figure setup
f = figure;
f.Position(3:4) = [1100,350];
t = tiledlayout(1,3, 'TileSpacing', 'Compact', 'Padding', 'Compact');
title(t, 'Particle Structure Speedup', 'FontWeight', 'bold')
xlabel(t, {'Number Particles (Ten Thousands)','Number Elements'}, 'FontWeight', 'bold')
ylabel(t, {'Average Structure Speedup','(SCS Time/Structure Time)'}, 'FontWeight', 'bold')

% Even (excluded)

cabm_length = structLen(CABM);
csr_length = structLen(CSR);
dps_length = structLen(DPS);

% Uniform
ax2 = nexttile;
semilogy( ...
    elms(1:cabm_length), scs_push_uni(1:cabm_length)./cabm_push_uni, 'r:', ...
    elms(1:cabm_length), scs_rebuild_uni(1:cabm_length)./cabm_rebuild_uni, 'r--', ...
    elms(1:cabm_length), scs_migrate_uni(1:cabm_length)./cabm_migrate_uni, 'r-.', ...
    elms(1:csr_length), scs_push_uni(1:csr_length)./csr_push_uni, 'b:', ...
    elms(1:csr_length), scs_rebuild_uni(1:csr_length)./csr_rebuild_uni, 'b--', ...
    elms(1:csr_length), scs_migrate_uni(1:csr_length)./csr_migrate_uni, 'b-.', ...
    elms(1:dps_length), scs_push_uni(1:dps_length)./dps_push_uni, 'g:', ...
    elms(1:dps_length), scs_rebuild_uni(1:dps_length)./dps_rebuild_uni, 'g--', ...
    elms(1:dps_length), scs_migrate_uni(1:dps_length)./dps_migrate_uni, 'g-.', ...
    elms, ones(size(elms)), 'k', ...
    'LineWidth', LineWidth );
ax = gca;
ax.XAxis.Exponent = 0;
ax.YTick = YTick;
ax.YTickLabel = YTickLabel;
ax.YGrid = 'on';
title({'Uniform Distribution'})
lim2 = axis;

% Gaussian
ax3 = nexttile;
semilogy( ...
    elms(1:cabm_length), scs_push_gauss(1:cabm_length)./cabm_push_gauss, 'r:', ...
    elms(1:cabm_length), scs_rebuild_gauss(1:cabm_length)./cabm_rebuild_gauss, 'r--', ...
    elms(1:cabm_length), scs_migrate_gauss(1:cabm_length)./cabm_migrate_gauss, 'r-.', ...
    elms(1:csr_length), scs_push_gauss(1:csr_length)./csr_push_gauss, 'b:', ...
    elms(1:csr_length), scs_rebuild_gauss(1:csr_length)./csr_rebuild_gauss, 'b--', ...
    elms(1:csr_length), scs_migrate_gauss(1:csr_length)./csr_migrate_gauss, 'b-.', ...
    elms(1:dps_length), scs_push_gauss(1:dps_length)./dps_push_gauss, 'g:', ...
    elms(1:dps_length), scs_rebuild_gauss(1:dps_length)./dps_rebuild_gauss, 'g--', ...
    elms(1:dps_length), scs_migrate_gauss(1:dps_length)./dps_migrate_gauss, 'g-.', ...
    elms, ones(size(elms)), 'k', ...
    'LineWidth', LineWidth );
ax = gca;
ax.XAxis.Exponent = 0;
ax.YTick = YTick;
ax.YTickLabel = YTickLabel;
ax.YGrid = 'on';
title({'Gaussian Distribution'})
lim3 = axis;

% Exponential
ax4 = nexttile;
semilogy( ...
    elms(1:cabm_length), scs_push_exp(1:cabm_length)./cabm_push_exp, 'r:', ...
    elms(1:cabm_length), scs_rebuild_exp(1:cabm_length)./cabm_rebuild_exp, 'r--', ...
    elms(1:cabm_length), scs_migrate_exp(1:cabm_length)./cabm_migrate_exp, 'r-.', ...
    elms(1:csr_length), scs_push_exp(1:csr_length)./csr_push_exp, 'b:', ...
    elms(1:csr_length), scs_rebuild_exp(1:csr_length)./csr_rebuild_exp, 'b--', ...
    elms(1:csr_length), scs_migrate_exp(1:csr_length)./csr_migrate_exp, 'b-.', ...
    elms(1:dps_length), scs_push_exp(1:dps_length)./dps_push_exp, 'g:', ...
    elms(1:dps_length), scs_rebuild_exp(1:dps_length)./dps_rebuild_exp, 'g--', ...
    elms(1:dps_length), scs_migrate_exp(1:dps_length)./dps_migrate_exp, 'g-.', ...
    elms, ones(size(elms)), 'k', ...
    'LineWidth', LineWidth );
ax = gca;
ax.XAxis.Exponent = 0;
ax.YTick = YTick;
ax.YTickLabel = YTickLabel;
ax.YGrid = 'on';
title({'Exponential Distribution'})
lim4 = axis;

lg = legend(nexttile(3), {'CabM pseudo-push'; 'CabM rebuild'; 'CabM migrate'; ...
    'CSR pseudo-push'; 'CSR rebuild'; 'CSR migrate'; ...
    'DPS pseudo-push'; 'DPS rebuild'; 'DPS migrate'; ...
    'SCS (reference)'}, 'FontWeight', 'bold');
lg.Location = 'northeastoutside';

% align axes
limits = [lim2; lim3; lim4];
limits = [ min(limits(:,1)), max(limits(:,2)), min(limits(:,3)), max(limits(:,4)) ];
axis([ax2 ax3 ax4], limits )

saveas(f,outname)
err = 0;
end



function data = readFileDataAndClose(fileId)
data = fscanf(fileId, "%d %d %d %f", [4 Inf])';
fclose(fileId);
return
end

function structLen = getStructureLengths(data)
structLen = zeros(1,3);
for i = 0:3
    structLen(i+1) = length(unique(data( data(:,1) == i, 2 )));
end
end

function elms = getElms(data, structLen)
maxLen = max(structLen);
if ( maxLen == structLen(1) )
    elms = unique(data( data(:,1) == 0, 2 ));
elseif ( maxLen == structLen(2) )
    elms = unique(data( data(:,1) == 1, 2 ));
elseif ( maxLen == structLen(3) )
    elms = unique(data( data(:,1) == 2, 2 ));
else
    elms = unique(data( data(:,1) == 3, 2 ));
end
end


function time = getFunctionTimeForStructure(data, structure)
time = data( data(:,1) == structure, [3,4]);
end

function time = getTime(data, structure, distribution)
time = data( data(:,1) == structure & data(:,3) == distribution, 4);
end

function time = getTime2(allData, func, structure, distribution)
data = allData{func};
time = data( data(:,1) == structure & data(:,3) == distribution, 4);
end

% TODO create the matrix and use this in the plotting functions instead of 
%      individual arrays
function mat = getMatrix(allData,NUMFUNC,NUMSTRUCT,NUMDIST,MAXELMS)
mat = zeros(NUMFUNC,NUMSTRUCT,NUMDIST,MAXELMS);
for func = 1:NUMFUNC
    for struct = 1:NUMSTRUCT
        for dist = 1:NUMDIST
            time = getTime2(allData, func, struct, dist);
            time2 = reshape(time, [1,1,1,MAXELMS])
            mat(func,struct,dist,:) = time2; % FIXME
        end
    end
end
end

