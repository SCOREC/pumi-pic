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
    elms(1:cabm_length), getTimeRatio(mat,PUSH,UNIFORM,SCS+1,CABM+1,cabm_length), 'r:', ...
    elms(1:cabm_length), getTimeRatio(mat,REBUILD,UNIFORM,SCS+1,CABM+1,cabm_length), 'r--', ...
    elms(1:cabm_length), getTimeRatio(mat,MIGRATE,UNIFORM,SCS+1,CABM+1,cabm_length), 'r-.', ...
    elms(1:csr_length), getTimeRatio(mat,PUSH,UNIFORM,SCS+1,CSR+1,csr_length), 'b:', ...
    elms(1:csr_length), getTimeRatio(mat,REBUILD,UNIFORM,SCS+1,CSR+1,csr_length), 'b--', ...
    elms(1:csr_length), getTimeRatio(mat,MIGRATE,UNIFORM,SCS+1,CSR+1,csr_length), 'b-.', ...
    elms(1:dps_length), getTimeRatio(mat,PUSH,UNIFORM,SCS+1,DPS+1,dps_length), 'g:', ...
    elms(1:dps_length), getTimeRatio(mat,REBUILD,UNIFORM,SCS+1,DPS+1,dps_length), 'g--', ...
    elms(1:dps_length), getTimeRatio(mat,MIGRATE,UNIFORM,SCS+1,DPS+1,dps_length), 'g-.', ...
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
    elms(1:cabm_length), getTimeRatio(mat,PUSH,GAUSS,SCS+1,CABM+1,cabm_length), 'r:', ...
    elms(1:cabm_length), getTimeRatio(mat,REBUILD,GAUSS,SCS+1,CABM+1,cabm_length), 'r--', ...
    elms(1:cabm_length), getTimeRatio(mat,MIGRATE,GAUSS,SCS+1,CABM+1,cabm_length), 'r-.', ...
    elms(1:csr_length), getTimeRatio(mat,PUSH,GAUSS,SCS+1,CSR+1,csr_length), 'b:', ...
    elms(1:csr_length), getTimeRatio(mat,REBUILD,GAUSS,SCS+1,CSR+1,csr_length), 'b--', ...
    elms(1:csr_length), getTimeRatio(mat,MIGRATE,GAUSS,SCS+1,CSR+1,csr_length), 'b-.', ...
    elms(1:dps_length), getTimeRatio(mat,PUSH,GAUSS,SCS+1,DPS+1,dps_length), 'g:', ...
    elms(1:dps_length), getTimeRatio(mat,REBUILD,GAUSS,SCS+1,DPS+1,dps_length), 'g--', ...
    elms(1:dps_length), getTimeRatio(mat,MIGRATE,GAUSS,SCS+1,DPS+1,dps_length), 'g-.', ...
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
    elms(1:cabm_length), getTimeRatio(mat,PUSH,EXPONENTIAL,SCS+1,CABM+1,cabm_length), 'r:', ...
    elms(1:cabm_length), getTimeRatio(mat,REBUILD,EXPONENTIAL,SCS+1,CABM+1,cabm_length), 'r--', ...
    elms(1:cabm_length), getTimeRatio(mat,MIGRATE,EXPONENTIAL,SCS+1,CABM+1,cabm_length), 'r-.', ...
    elms(1:csr_length), getTimeRatio(mat,PUSH,EXPONENTIAL,SCS+1,CSR+1,csr_length), 'b:', ...
    elms(1:csr_length), getTimeRatio(mat,REBUILD,EXPONENTIAL,SCS+1,CSR+1,csr_length), 'b--', ...
    elms(1:csr_length), getTimeRatio(mat,MIGRATE,EXPONENTIAL,SCS+1,CSR+1,csr_length), 'b-.', ...
    elms(1:dps_length), getTimeRatio(mat,PUSH,EXPONENTIAL,SCS+1,DPS+1,dps_length), 'g:', ...
    elms(1:dps_length), getTimeRatio(mat,REBUILD,EXPONENTIAL,SCS+1,DPS+1,dps_length), 'g--', ...
    elms(1:dps_length), getTimeRatio(mat,MIGRATE,EXPONENTIAL,SCS+1,DPS+1,dps_length), 'g-.', ...
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
    for struct = 0:NUMSTRUCT-1
        structIdx = struct+1;
        for dist = 1:NUMDIST-1 %skipping EVEN, zero based indexing -> don't include end
            time = getTime2(allData, func, struct, dist);
            assert( isempty(time) == false );
            for elm = 1:MAXELMS
                mat(func,structIdx,dist,elm) = time(elm);
            end
        end
    end
end
end

function ratio = getTimeRatio(mat,func,dist,structA,structB,structB_length)
ratio = squeeze(mat(func,structA,dist,1:structB_length)./mat(func,structB,dist,:));
end
