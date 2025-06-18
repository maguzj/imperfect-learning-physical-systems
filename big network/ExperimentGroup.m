
classdef ExperimentGroup < network_project_superclass2
    % Class instance for a unique physical coupled learning network.
    
    %% Properties
    properties (Constant)
        
        ObjName = 'group';
        SaveFolder = network_project_superclass2.GroupFolder;
    end
    
    properties (SetAccess = immutable) %
        Name
        
        
        ERRTHRESH = 10; % multiple of final error value where plateau starts
    end
    
    properties
        
        DIR
        ADDSTR
        
        DATENUM
        
        % Scalars
        SOR % number of sources
        TAR % number of targets
        EPO % number of epochs
        TRA % training set size
        TST % test set size
        ETA % nudge amp*129
        ALF % learning time (in us)
        REP % measurement repetitions
        MES % number of measurements that will be taken
        RES % number of on-off cycles to edge reset performed at the start
        VMN % minimum voltage value (Vmin ~ 8*VMN/129)
        VIT % initialization voltage (Vinit ~ 8*VIT/129)
        CLA % number of classes (1 ignores classes)
        SPR %  rate class bases spread from mean value (multiplied by 1+SPR/1000 times themselves, - SPR/1000*mean)
        VEG % number of vertical edges in each column (4 is periodic)
        HEG % number of horizontal edges in each row (4 is periodic)
        DEL
        UPT
        UPD
        BUF
        HOT
        ANT
        DUP
        BTH
        LOK
        REG
        GRU
        SNO
        TNO
        NOR
        AML
        MNA
        
        DYNAMICETA
        
        GATEMULT
        NODEMULT
        RCHARGE
        RDRAIN
        CCHARGE
        ROFF
        SOURCERES
        DIODES
        
        AMP
        NUMVERTEDGE
        NUMHORIZEDGE
        TDIST
        MAXT
        MINT
        TRAINTIME1
        TRAINTIMEE4
        T1ROW
        T1COL
        T1NODE
        MAXS1
        CAPMIN
        CAPMAX
        CAPSTART
        NEDGES
        TRAINSET
        TESTSET
        TMAT
        ALFETA
        PREVEXP
        MEASCONS
        TRAINIDX
        VARINP % count of variable inputs
        
        
        ERR1TRAIN
        ERR1TRAINING
        ERR1
        ERR2
        PWR
        ERR1STD
        ERR2STD
        PWRSTD
        XORGAP
        XORGAPTRAIN
        XORGAPMAX
        CLASSERR
        CLASSERRTRAIN
        CLASSERRMAX
        CLASSERRPERF
        CLASSERRPERFTRAIN
        DISCSUM
        DISCMAX
        ERR1BEST
        AVGCAP
        MINCAP
        MAXCAP
        CAPEND
        MODE0 % mean error
        MODE1 %
        MODE2
        DYNERR
        DELTAG
        DELTAK
        DELTAGBIAS
        DELTAERRTRAIN
        HIGHWALLS
        LOWWALLS
        HINGEERR
        HINGEERRTRAIN
        DELTAHINGETRAIN
        GERR
        
        TRAINTIME
        
        
        PROBLEMCAPS
        
        PARAMNAMES =[...
            "RCHARGE","RDRAIN","CCHARGE","ROFF",...
            "EPO", "TRA", "TST", "ETA", ...
            "ALF", "SOR", "TAR","REP","MES",...
            "RES","VMN","CLA","SPR","VIT","VEG","HEG","DEL",...
            "GATEMULT","NODEMULT","SOURCERES","UPT","UPD","BUF","HOT","ANT",...
            "DUP","LOK","BTH","REG","GRU","SNO","TNO","NOR","AML","MNA"];
        
        CALCNAMES = [...
            "AMP","NUMVERTEDGE","NUMHORIZEDGE","TDIST",...
            "MAXT","MINT","TRAINTIME1","T1ROW","T1COL","T1NODE","MAXS1",...
            "CAPMIN","CAPMAX","CAPSTART","NEDGES","TRAINSET","ALFETA",...
            "ERR1BEST","PREVEXP","MEASCONS","DIODES","TRAINIDX","TRAINTIMEE4",...
            "TESTSET","VARINP","DYNAMICETA","TRAINTIME","TMAT","CLASSERRPERF","CLASSERRPERFTRAIN"
            ];
        
        SPOTNAMES = [...,
            "ERR1","ERR2","ERR1STD",...
            "ERR2STD","PWRSTD","PWR","XORGAP","XORGAPTRAIN"...
            "CLASSERR","CLASSERRTRAIN","CLASSERRMAX",...
            "ERR1TRAIN","DISCSUM","DISCMAX","AVGCAP",...
            "MINCAP","MAXCAP","XORGAPMAX","MODE0","MODE1","MODE2","ERR1TRAINING","DYNERR","DELTAG","DELTAK",...
            "HIGHWALLS","LOWWALLS", "HINGEERR","HINGEERRTRAIN","DELTAGBIAS","CAPEND","DELTAERRTRAIN","DELTAHINGETRAIN"];
        
        TAGS = ["ERR1","ERR2","ERR1STD",...
            "ERR2STD","PWRSTD","PWR","ALF","VIT","VMN"...
            ,"TRAINTIME1","ETA",...
            "RCHARGE","RDRAIN","ROFF","SOURCERES",...
            "CAPMIN","CAPMAX","CAPSTART","ERR1TRAIN",...
            "AMP","DISCSUM","DISCMAX",...
            "ERR1BEST","AVGCAP","MINCAP","MAXCAP","TRAINTIMEE4",...
            "ERR1TRAINING","DYNERR","DELTAG","DELTAK", "HINGEERR","HINGEERRTRAIN","DELTAGBIAS","GERR",...
            "DELTAERRTRAIN","DELTAHINGETRAIN"];
        UNITS = [ "V^2", "V^2", "V^2",...
            "V^2","mW","mW","\muS","V","V",...
            "mS","",...
            "\Omega","\Omega","\Omega","\Omega",...
            "V","V","V","V^2",...
            "","V^2","V^2",...
            "V^2","V","V","V","mS",...
            "V^2","V^2","V","V", "V^2", "V^2","V","V^2",...
            "V^2","V^2"];
        MULTS = [1,1,1,...
            1,1000,1000,1,8/129,8/129,...
            1/1000,1/129,...
            1,1,1,1,...
            1,1,1,1,...
            1,1,1,...
            1,1,1,1,1/1000,...
            1,1,1,1,1,1,1,1,...
            1,1];
        
        SHIFTS = [0,0,0,...
            0,0,0,0,-.1,-.37,...
            0,0,...
            0,0,0,0,...
            0,0,0,0,...
            -1,0,0,...
            0,0,0,0,1,...
            0,0,0,0,0,0,0,0,...
            0,0];
        
        numfigs = 0;
        openfigs = {};
        trainsetcell = {};
        testsetcell = {};
        tmatcell = {};
    end
    
    
    %% Methods
    methods
        
        
        
        % Create an instance of Experiment2. All inputs required.
        function EG = ExperimentGroup(Wantstr,Wantvec,expfldrnum,ONLYTWOEDGES)
            
            EG@network_project_superclass2();
            EG.FullPathSaveName = EG.fullFilename();
            
            if nargin<2
                error('Not enough inputs: need Name, Wantstr (to match experiment names), and Wantvec (for error/power calculation).')
            end
            
            if sum(Wantvec>0)*sum(Wantvec<0)
                error('Cannot have both positive and negative indicees in Wantvec!');
            end
            
            if ischar(Wantstr) || iscell(Wantstr)
                if ischar(Wantstr)
                    Wantstr = {Wantstr};
                end
                
                EG.Name = strcat(Wantstr{:});
                
                if nargin<3 || (~isequal(expfldrnum,2)&& ~isstr(expfldrnum))
                    EG.ADDSTR = '';
                elseif expfldrnum == 2
                    EG.ADDSTR = '/2022/';
                elseif isstr(expfldrnum)
                    EG.ADDSTR = expfldrnum;
                end
                
                if nargin<4 || isempty(ONLYTWOEDGES)
                    ONLYTWOEDGES = false;
                elseif length(ONLYTWOEDGES)==2
                    correctG = ONLYTWOEDGES;
                    ONLYTWOEDGES = true;
                end
                
                Wantvec = sort(Wantvec);
                
                
                %% Find all experiments that match this Wantstr
                fldr = [network_project_superclass2.ExperimentFolder, EG.ADDSTR];
                
                d = dir(fldr);
                want = false(size(d));
                for l = 1:length(Wantstr)
                    L = length(Wantstr{l});
                    for k = 1:length(d)
                        name = d(k).name;
                        if length(name)>(L+11)
                            if strcmp(name(11+(1:L)),Wantstr{l})
                                want(k) = true;
                                
                            end
                        end
                    end
                end
                
                % Eliminate those that do not match and sort remaining by name/date
                d = d(want);
                vals = cell(1,length(d));
                for k = 1:length(d)
                    vals{k} = d(k).name;
                end
                [~,ind]=sort(vals);
                EG.DIR = d(ind);
                EG.DATENUM = [d.datenum];
                
            elseif isstruct(Wantstr)
                EG.Name = '$Struct Input$';
                EG.DIR = Wantstr;
            end
            
            %% Load data from each experiment, eliminate if error
            SCALARNAMES = [EG.PARAMNAMES,EG.CALCNAMES,EG.SPOTNAMES];
            N = length(EG.DIR);
            for k = 1:length(SCALARNAMES)
                EG.(SCALARNAMES(k)) = nan(1,N);
            end
            
            stillwant = false(1,N);
            
            W = waitbar(0,'Loading Experimental Data');
            
            for k = 1:N
                % disp(k)
                waitbar(k/N,W);
                
                try
                    load([EG.DIR(k).folder,'\',EG.DIR(k).name],'experiment')
                catch
                    
                    disp(['Error loading file ',EG.DIR(k).name])
                    continue
                end
                if isempty(experiment.TestMSE) && experiment.TST>0
                    continue % not completed!
                end
                
                if isempty(experiment.TestMSE2) || isempty(experiment.TrainingMSE)
                    try
                        experiment = experiment.CalcError();
                        experiment.save();
                    catch
                        disp(['error calculating error for ',experiment.fullFilename])
                        continue
                    end
                end
                
                if max(abs(Wantvec))>experiment.MES || isempty(experiment.TrainFreeState)
                    continue
                end
                
                %% Load PARAMNAMES quantities
                
                for n = 1:length(EG.PARAMNAMES)
                    EG.(EG.PARAMNAMES(n))(k) = experiment.(EG.PARAMNAMES(n));
                end
                
                
                
                %% Calculate SPOTNAMES quantities
                if Wantvec(1)<=0 % if counting backwards
                    avgvec = size(experiment.TestMSE,2) + Wantvec;
                else
                    avgvec = Wantvec;
                end
                try
                    err0 = sum(experiment.TestMSE,1)*(experiment.NODEMULT^2);
                    errbest = min(err0);
                    err = err0(avgvec);
                catch
                    disp('Error calculating test MSE')
                    continue
                end
                err0train = sum(experiment.TrainMSE,1)*(experiment.NODEMULT^2);
                errtrain = err0train(avgvec)*(experiment.NODEMULT^2);
                
                
                
                if ~isempty(experiment.TARGFLAG)
                    try
                        [dynerr,trainvecidx] = experiment.trainingError(avgvec);
                        dynerr = dynerr(logical(experiment.TARGFLAG(trainvecidx)));
                        EG.DYNERR(k) = mean(dynerr);
                    catch
                    end
                end
                
                err0training = sum(experiment.TrainingMSE,1)*(experiment.NODEMULT^2);
                errtraining = err0training(avgvec);
                
                pwr = mean(experiment.calc_state_power(avgvec),2);
                
                stvals = experiment.nameInputs();
                
                try
                    [modeerrors,orders] = modeErrors(experiment,avgvec,false,2);
                    
                    for kk = 0:2
                        wantO = find(sum(orders,2)==kk);
                        allwant = abs(modeerrors(wantO,:));
                        
                        switch kk
                            case 0
                                EG.MODE0(k) = mean(allwant(:));
                            case 1
                                EG.MODE1(k) = mean(allwant(:));
                            case 2
                                EG.MODE2(k) = mean(allwant(:));
                        end
                    end
                catch
                    % disp('modeErrors error in ExperimentGroup.m');
                end
                
                [bot,top] = experiment.capsOnWalls(avgvec);
                EG.HIGHWALLS(k) = mean(top);
                EG.LOWWALLS(k) = mean(bot);
                
                
                EG.ERR1BEST(k) = errbest;
                EG.ERR1TRAIN(k) = mean(errtrain);
                EG.ERR1TRAINING(k) = mean(errtraining);
                EG.ERR1(k) = mean(err);
                EG.ERR1STD(k) = std(err);
                EG.PWR(k) = mean(pwr);
                EG.PWRSTD(k) = std(pwr);
                
                caps = experiment.capacitors(avgvec);
                
                EG.AVGCAP(k) = nanmean(caps(:));
                EG.MINCAP(k) = nanmin(caps(:));
                EG.MAXCAP(k) = nanmax(caps(:));
                
                TAU = -experiment.TRAINIDX(1);
                try
                    if experiment.GRU > 1
                        TAU = TAU*experiment.TRA/experiment.GRU;
                    end
                catch
                end
                
                  if experiment.CLA>1
                    [outTest,outTrain] = experiment.ClassificationError();
                    EG.CLASSERR(k) = mean(outTest(avgvec));
                    EG.CLASSERRTRAIN(k) = mean(outTrain(avgvec));
                    EG.CLASSERRMAX(k) = max(outTest);
                    EG.CLASSERRPERF(k) = mean(outTest(avgvec)==0);
                    EG.CLASSERRPERFTRAIN(k) = mean(outTrain(avgvec)==0);
                    
                    [~,~,~,~,~,hingeerr,hingeerrTRAIN] = experiment.FoundClassIDs(); % false = no BUF
                    EG.HINGEERR(k) = mean(mean(hingeerr(avgvec,:).^2))*(experiment.NODEMULT^2);
                    meanhingeTRAIN = mean(hingeerrTRAIN.^2,2)*(experiment.NODEMULT^2);
                    EG.HINGEERRTRAIN(k) = mean(meanhingeTRAIN(avgvec));
                    
                 
                    
                end
                
                
                if length(unique(experiment.TRAINIDX))==1 && TAU>0
                    ct = 0;
                    trainsteps = (experiment.DOTEST(avgvec)-1);
                    keepsteps = trainsteps == round(trainsteps/TAU)*TAU;
                    avgvec2 = avgvec(keepsteps);
                    
                    deltak = 0;
                    deltag = 0;
                    deltaerr = 0;
                    deltahinge = 0;
                    g1 = 0;
                    g2 = 0;
                    if exist('correctG')
                        g1err = 0;
                        g2err = 0;
                    end
                    usedvals = [];
                    
                    for ks = avgvec2
                        pairidx = find(experiment.DOTEST==(experiment.DOTEST(ks)+TAU),1);
                        
                        % no double counting -- bad for bias calculation
                        if ~isempty(pairidx)
                            
                            if rem(experiment.DOTEST(ks),TAU*2)==1 %keep first one always same datapoint
                                pair = [pairidx,ks];
                            else
                                pair = [ks,pairidx];
                            end
                            
                            if (isempty(find(usedvals==pair(1),1)) &&  isempty(find(usedvals==pair(2),1)))
                                
                                
                                usedvals = [usedvals,pair];
                             
                                deltag = deltag + diff(experiment.capacitors(pair),1,3).^2;
                                deltaerr = deltaerr + sum(err0train(pair))/2;
                                if experiment.CLA>1
                                    deltahinge = deltahinge + mean(meanhingeTRAIN(pair));%mean(mean(hingeerrTRAIN(pair,:).^2))*(experiment.NODEMULT^2);
                                end
                                deltak = deltak + diff(experiment.state_map(pair,0),1,3).^2;
                                ct = ct+1;
                                
                                g1 = g1+experiment.capacitors(pair(1));
                                g2 = g2+experiment.capacitors(pair(2));
                                
                                
                            end
                        end
                    end
                    
                    if numel(deltag) == 1
                        disp('finding last pair of valid deltaGs')
                        
                        pairidx = [];
                        ks = experiment.MES;
                        while isempty(pairidx) && ks > 10
                            
                            if  (experiment.DOTEST(ks)-1) == round((experiment.DOTEST(ks)-1)/TAU)*TAU
                                pairidx = find(experiment.DOTEST==(experiment.DOTEST(ks)+TAU),1);
                            end
                            ks = ks-1;
                        end
                        
                        deltag = deltag + diff(experiment.capacitors([ks,pairidx]),1,3).^2;
                        deltak = deltak + diff(experiment.state_map([ks,pairidx],0).^-1,1,3).^2;
                        deltaerr = deltaerr + sum(err0train([ks,pairidx]))/2;
                        if experiment.CLA>1
                            deltahinge = deltahinge + mean(meanhingeTRAIN([ks,pairidx]));%mean(mean(hingeerrTRAIN(pair,:).^2))*(experiment.NODEMULT^2);
                        end
                        
                        ct = ct+1;
                    end
                    
                    if ONLYTWOEDGES
                        try
                            deltag = deltag(10,[5,9],:);
                            deltak = deltak(10,[5,9],:);
                            g1 = g1(10,[5,9],:);
                            g2 = g2(10,[5,9],:);
                            disp('ONLY LOOKING AT TWO EDGES RN')
                            
                        catch
                            continue
                        end
                    end
                    
                    g1 = g1/ct;
                    g2 = g2/ct;
                    deltaerr = deltaerr/ct;
                    deltahinge = deltahinge/ct;
                    deltag = nansum(nansum(deltag,1),2)/ct;
                    deltak = nansum(nansum(deltak,1),2)/ct;
                    
                    deltagbias = (nansum(nansum(((g1-g2).^2),1),2));
                    try
                        if ~isempty(deltag)
                        EG.DELTAG(k) = sqrt(deltag);
                        EG.DELTAK(k) = sqrt(deltak);
                        EG.DELTAGBIAS(k) = sqrt(deltagbias);
                        EG.DELTAERRTRAIN(k) = deltaerr;
                        EG.DELTAHINGETRAIN(k) = deltahinge;
                        else
                            
                         EG.DELTAG(k) = nan;
                        EG.DELTAK(k) = nan;
                        EG.DELTAGBIAS(k) = nan;
                        EG.DELTAERRTRAIN(k) = nan;
                        EG.DELTAHINGETRAIN(k) = nan;
                        end
                    catch
                        disp('something wrong!')
                        
                        continue
                    end
                    
                    if exist('correctG')
                        disp('correctG dist calculated assuming no noise')
                        EG.GERR(k) = sum((g1-correctG).^2)+sum((g2-correctG).^2);
                    end
                    
                    
                    
                end
                
                [DS,DM] = experiment.discrepancySum(avgvec);
                EG.DISCSUM(k) = mean(DS(:));
                EG.DISCMAX(k) = mean(DM(:));
                
                try
                    x2 = xor_quality2(experiment,avgvec);
                    EG.XORGAP(k) = mean(x2(:)); % test
                    EG.XORGAPTRAIN(k) =nan;
                    EG.XORGAPMAX(k) = max(x2(:));
                    
                catch
                    
                end
                
              
                
                %% Calculate CALCNAMES quantities
                if isempty(experiment.NSIZE2)
                    experiment.NSIZE2 = experiment.NSIZE;
                end
                if isempty(experiment.ISPERIODIC)
                    experiment.ISPERIODIC = [1,1];
                end
                
                EG.AMP(k) = experiment.amp_ratio();
                EG.NUMVERTEDGE(k) = experiment.NSIZE2(1) + experiment.ISPERIODIC(1);
                EG.NUMHORIZEDGE(k) = experiment.NSIZE2(2) + experiment.ISPERIODIC(2);
                EG.MAXT(k) = max(max(experiment.TRAIN((experiment.SOR+1):end,:)));
                EG.MINT(k) = min(min(experiment.TRAIN((experiment.SOR+1):end,:)));
                EG.MAXS1(k) = max(max(experiment.TRAIN(1,:)));
                
                EG.TRAINTIME(k) = sum(experiment.LearnTimes);
                idx = find(or(err0<EG.ERR1(k)/EG.ERRTHRESH,err0>(EG.ERR1(k)*EG.ERRTHRESH)),1,'last');
                EG.TRAINTIME1(k) = sum(experiment.LearnTimes(1:idx));
                idx2 = find(err0>10^-4,1,'last');
                if isempty(idx2)
                    EG.TRAINTIMEE4(k) = nan;
                else
                    EG.TRAINTIMEE4(k) = sum(experiment.LearnTimes(1:idx2));
                end
                
                EG.T1ROW(k) = experiment.TLOC(1,1);
                EG.T1COL(k) = experiment.TLOC(1,2);
                EG.T1NODE(k) = EG.T1ROW(k)*4 + EG.T1COL(k);
                caps = experiment.capacitors([1,-1]);
                caps1 = caps(:,:,1);
                capsend = caps(:,:,2);
                EG.CAPSTART(k) = nanmean(caps1(:));
                EG.CAPEND(k) = nanmean(capsend(:));
                EG.CAPMIN(k) = nanmin(capsend(:));
                EG.CAPMAX(k) = nanmax(capsend(:));
                EG.NEDGES(k) = EG.VEG(k)+EG.HEG(k)/10;
                
                EG.MEASCONS(k) = experiment.measurementConsistency();
                EG.DIODES(k) = numel(experiment.DIODES)/4;
                EG.TRAINIDX(k) = mode(experiment.TRAINIDX);
                
                EG.DYNAMICETA(k) = length(unique(experiment.ETAS))>1;
                
                EG.VARINP(k) = sum(imag(stvals)~=0);
                
                if experiment.CLA>1 && experiment.UPD ~= 5
                    
                    TRAIN= [experiment.TRAIN(1:experiment.SOR,:);experiment.TRAINCLASSES];
                    if experiment.TST == 0
                        TEST = [];
                    else
                    TEST= [experiment.TEST(1:experiment.SOR,:);experiment.TESTCLASSES];
                    end
                else
                    TRAIN= experiment.TRAIN;
                    if experiment.TST == 0
                        TEST = [];
                    else
                    TEST= experiment.TEST;
                    end
                end
                TMAT = experiment.TMAT;
                
                
                if k == 1
                    EG.TRAINSET(k) = 1;
                    EG.trainsetcell{1} = TRAIN;
                    EG.TESTSET(k) = 1;
                    EG.testsetcell{1} = TEST;
                    EG.tmatcell{end+1} = TMAT;
                    EG.TMAT(k) = length(EG.tmatcell);
                else
                    found = false;
                    
                    for t = unique(EG.TRAINSET(~isnan(EG.TRAINSET)))
                        
                        if isequal(EG.trainsetcell{t},TRAIN)
                            EG.TRAINSET(k) = t;
                            found = true;
                            break
                        end
                    end
                    if ~found
                        EG.trainsetcell{end+1} = TRAIN;
                        EG.TRAINSET(k) = length(EG.trainsetcell);
                    end
                    
                    found = false;
                    
                    for t = unique(EG.TESTSET(~isnan(EG.TESTSET)))
                        
                        if isequal(EG.testsetcell{t},TEST)
                            EG.TESTSET(k) = t;
                            found = true;
                            break
                        end
                    end
                    if ~found
                        EG.testsetcell{end+1} = TEST;
                        EG.TESTSET(k) = length(EG.testsetcell);
                    end
                    
                    found = false;
                    
                    for t = unique(EG.TMAT(~isnan(EG.TMAT)))
                        
                        if isequal(EG.tmatcell{t},TMAT)
                            EG.TMAT(k) = t;
                            found = true;
                            break
                        end
                    end
                    if ~found
                        EG.tmatcell{end+1} = TMAT;
                        EG.TMAT(k) = length(EG.tmatcell);
                    end
                end
                
                
                EG.NEDGES(k) = EG.VEG(k)+EG.HEG(k)/10;
                EG.ALFETA(k) = EG.ALF(k)*EG.ETA(k);
                
                
                
                
                % Calculate mean source to target distance
                dists = zeros(experiment.SOR,experiment.TAR);
                for sor = 1:experiment.SOR
                    for tar = 1:experiment.TAR
                        dist = abs([experiment.TLOC(tar,1)-experiment.SLOC(sor,1),...
                            experiment.TLOC(tar,2)-experiment.SLOC(sor,2)]);
                        
                        for per = 1:2
                            if experiment.ISPERIODIC(per)
                                dist(per) = min(dist(per),experiment.NSIZE2(per)-dist(per));
                            end
                        end
                        dists(sor,tar) = sum(dist);
                    end
                end
                
                EG.TDIST(k) = mean(dists(:));
                
                
                stillwant(k) = true;
                
                
            end
            close(W)
            
            %% Remove any unfinished experiments
            for n = 1:length(SCALARNAMES)
                EG.(SCALARNAMES(n)) = EG.(SCALARNAMES(n))(stillwant);
            end
            EG.DIR = EG.DIR(stillwant);
            
        end % End of creator function
        
        % Creates figure with number unique to this ExperimentGroup (based
        % on date of creation)
        function EG = newFig(EG)
            EG.numfigs = EG.numfigs + 1;
            h = figure(EG.fignum(EG.numfigs));
            EG.openfigs{end+1} = h;
            if nargout == 0
                warning('ExperimentGroup not output of newFig -- handle not stored.')
            end
        end
        
        % returns figure number for this object (appended with inputnum)
        function num = fignum(EG,inputnum)
            num = str2double([erase(EG.Date(6:end),'-'),num2str(inputnum)]);
        end
        
        % closes all associated figures
        function EG = closeFigs(EG)
            if nargout == 0
                error('ExperimentGroup not output of closeFigs.')
            end
            
            for k = 1:max(100,EG.numfigs)
                try
                    close(EG.fignum(k))
                catch
                end
            end
            
            EG.numfigs = 0;
            EG.openfigs = {};
        end
        
        % given a struct W with arrays of parameters values (e.g.
        % W.ALF=[5,10]), output a list same size as lists (e.g. EG.ERR1)
        % constructed matching all parameters. Can match any parameter
        % within a given list but must match one from each list.
        function want =  getWant(EG,W)
            want = true(size(EG.ERR1));
            names = fieldnames(W);
            for k = 1:length(names)
                nme = names{k};
                if ~isempty(W.(nme))
                    want0 = false;
                    for idx = 1:length(W.(nme))
                        want0 = or(want0,W.(nme)(idx)==EG.(nme));
                    end
                    
                    want = and(want,want0);
                end
            end
            
        end
        
        function out = get(EG,name)
            name = upper(name);
            out = EG.(name)*EG.findMult(name)+EG.findShift(name);
        end
        
        function out = compoundValue(EG,expn)
            % Returns the output of the expression in 'expn', calculated using  specified properties of EG. For example: 'ERRBEST-ERR1' returns EG.ERRBEST-EG.ERR1.
            
            wantchar = '/*^()+-';
            outstr ='';
            startidx = 1;
            for idx = 1:length(expn)
                k = strfind(wantchar,expn(idx));
                if ~isempty(k)
                    if startidx ~= idx % field name and then single character
                        outstr = [outstr,'EG.get("',expn(startidx:(idx-1)),'")'];
                    end
                    if k<=3
                        str = '.';
                    else
                        str = '';
                    end
                    %add single character
                    outstr = [outstr,str,expn(idx)];
                    startidx =idx+1;
                end
                
            end
            if startidx~=idx
                outstr = [outstr,'EG.get("',expn(startidx:idx),'")'];
            end
            out = eval(outstr);
            
        end
        
        % Creates figure with 3 plots: xname vs yname, and sortname vs
        % x/yname. In the first plot a scatter of data is shown color-coded
        % by sortname, with geometric averages and stds overlaid. The othe
        % two plots only have these averages.
        function errorbarPlot(EG,xname,yname,sortname,want,logplot,sterror)
            
            
            if nargin<6
                logplot = true;
            end
            
            if nargin<7
                sterror = false;
            end
            
            xname = upper(xname);
            yname = upper(yname);
            sortname = upper(sortname);
            
            % Select appropriate fields
            xvals = EG.compoundValue(xname);
            yvals = EG.compoundValue(yname);
            sortvals = EG.compoundValue(sortname);
            
            % filter values if requested
            if nargin >=5 && ~isempty(want)
                if isstruct(want)
                    
                    want = EG.getWant(want);
                end
                xvals = xvals(want);
                yvals = yvals(want);
                sortvals = sortvals(want);
                
            end
            
            if logplot
                disp('doing loglog and geometric avg')
                addstr = 'log_{10} ';
            else
                addstr = '';
            end
            
            % Generate unit text for labels/legends
            xname= [addstr,xname,EG.findUnit(xname)];
            yname= [addstr,yname,EG.findUnit(yname)];
            sortname= [sortname,EG.findUnit(sortname)];
            
            %                         div = sqrt(numel(newx));
            div = 1;
            
            shps = 'os';
            figure(22091231)
            clf
            leg = {};
            SV = unique(sortvals);
            for dummy = 1:2
                for val0 = 1:length(SV)
                    val = SV(val0);
                    
                    shp = shps(rem(val0,length(shps))+1);
                    
                    c0 = (find(val==unique(sortvals),1)-1)/(length(unique(sortvals))-1);
                    if isnan(c0)
                        c0 = 1;
                    end
                    c = [1-c0, c0 ,1];
                    want2 = sortvals == val;
                    if logplot
                        want2 = and(xvals>0,want2);
                        want2 = and(yvals>0,want2);
                    end
                    newx = xvals(want2);
                    newy = yvals(want2);
                    if logplot
                        newx = log10(newx);
                        newy = log10(newy);
                    end
                    
                    subplot(2,2,[1,3])
                    
                    
                    if sterror
                        div = sqrt(numel(newx));
                    end
                    
                    if dummy ==1
                        %     h = errorbar(newx, newy,...
                        %         xvalstd(want2),xvalstd(want2),...
                        %         yvalstd(want2),yvalstd(want2),'o','Color',0.5*ones(1,3),...
                        %         'MarkerFaceColor',c,'MarkerSize',10,...
                        %         'LineWidth',1);
                        h = plot(newx, newy,shp,...
                            'Color',0.5*ones(1,3),...
                            'MarkerFaceColor',c*0.5 + 0.5,'MarkerSize',10,...
                            'LineWidth',1);
                        try
                            h.HandleVisibility = 'off';
                        catch
                        end
                        hold on
                        
                        
                        subplot(2,2,[2])
                        errorbar(val,mean(newx),...
                            std(newx)/div,...
                            ['k',shp],'MarkerFaceColor',c,'MarkerSize',15,...
                            'LineWidth',2)
                        hold on
                        xlabel(sortname)
                        ylabel(xname)
                        if logplot
                            set(gca,'XScale','log')
                        end
                        
                        subplot(2,2,[4])
                        errorbar(val,mean(newy),...
                            std(newy)/div,...
                            ['k',shp],'MarkerFaceColor',c,'MarkerSize',15,...
                            'LineWidth',2)
                        xlabel(sortname)
                        ylabel(yname)
                        hold on
                        if logplot
                            set(gca,'XScale','log')
                        end
                        
                    else
                        
                        
                        
                        
                        errorbar(mean(newx),mean(newy),...
                            std(newy)/div,std(newy)/div,...
                            std(newx)/div,std(newx)/div,...
                            ['k',shp],'MarkerFaceColor',c,'MarkerSize',15,...
                            'LineWidth',2)
                        hold on
                        leg{end+1} = [sortname,' = ',num2str(round(val,3))];
                        
                    end
                    
                    
                end
            end
            
            subplot(2,2,[1,3])
            
            %set(gca,'YScale','log','XScale','log')
            
            xlabel(xname)
            ylabel(yname)
            legend(leg)
            makePlotPrettyNow(12)
            
            subplot(2,2,[2])
            makePlotPrettyNow(12)
            
            subplot(2,2,[4])
            makePlotPrettyNow(12)
            
            drawnow
            
        end
        
        % Loads a given experiment from index
        function experiment = loadExperiment(EG,k)
            if k <0
                k = length(EG.DIR)+k;
            end
            load([network_project_superclass2.ExperimentFolder,EG.ADDSTR,EG.DIR(k).name],'experiment');
        end
        
        
        
        function out = capacitorsOverTime(EG,want,numtimes,vec,NOFIG)
            
            STYLE = 3;
            
            if nargin<4 || isempty(vec)
                vec = linspace(0,7,61);
            end
            if nargin<3 || isempty(numtimes)
                numtimes = 5;
            end
            if nargin>=2 && ~isempty(want)
                d = EG.DIR(want);
            else
                d = EG.DIR;
            end
            
            out = zeros(numtimes,length(vec));
            
            W = waitbar(0,'Loading Capacitor Values');
            for k = 1:length(d)
                waitbar(k/length(d),W);
                matinfo = load([network_project_superclass2.ExperimentFolder,EG.ADDSTR,d(k).name],'experiment');
                out = out + matinfo.experiment.capacitorHistogram(vec,numtimes);
            end
            close(W);
            out = out/length(d);
            
            if nargin<5 || ~NOFIG
                figure(2209271)
                clf
                
                step= diff(vec(1:2))*(numtimes-1)/numtimes;
                
                
                leg = {};
                for t0 = 1:numtimes
                    col = ((t0-1)/(numtimes-1)) * [0,1,-1] + [0,0,1];
                    if STYLE == 1
                        bar(vec-step/2+col(2)*step,out(t0,:),'LineWidth',2,'EdgeColor',col,...
                            'FaceColor','none','BarWidth',1/numtimes)
                        
                    elseif STYLE == 2
                        plot(vec,out(t0,:),'-','LineWidth',2,'Color',col)
                    elseif STYLE == 3
                        plot3(vec,t0+vec*0,out(t0,:),'-','LineWidth',2,'Color',col)
                    end
                    hold on
                    %leg{end+1} = [num2str(round(time(timesteps(t0)))),'ms'];
                end
                xlabel('Capacitor Values (V)')
                
                if STYLE == 4
                    imagesc(log(out));
                    wantx = round(linspace(1,length(vec),10));
                    wanty = round(linspace(1,numtimes,5));
                    set(gca,'XTICK',wantx,'XTICKLABEL',round(vec(wantx),1),'YTICK',wanty);
                    
                end
                % legend(leg)
                if STYLE == 3 || STYLE == 4
                    ylabel('Time (AU)')
                    zlabel('Count')
                else
                    ylabel('Count')
                end
                
                if STYLE == 4
                    axis tight;
                end
                
            end
            
        end
        
        
        
        
        % create and plot average capacitor values at time index idx,
        % for all experiments in want, grouped by sortvar.
        function out = capacitorMap(EG,typeflag,idx,sortvar,want,NOFIG)
            
            try
                typeflag = typeflag{1};
            end
            
            if nargin<3
                idx = -1;
            end
            
            if length(idx)~=1
                error('idx must be scalar')
            end
            
            if nargin<4 || isempty(sortvar)
                sortvar = ones(size(EG.ERR1));
            end
            if nargin>=5 && ~isempty(want)
                d = EG.DIR(want);
                sortvar = sortvar(want);
            else
                d = EG.DIR;
            end
            
            SV = unique(sortvar);
            counts = zeros(size(SV));
            for s = 1:length(SV)
                counts(s) = sum(SV(s)==sortvar);
            end
            badflag = false;
            divon = badflag;
            
            if nargin<2 || strcmp(upper(typeflag),"MEAN") || strcmp(upper(typeflag),"AVG")
                fcn = @(a,b) a+b;
                out = zeros(12,12,length(SV));
                divon = true;
                
            elseif strcmp(upper(typeflag),"MAX")
                fcn = @(a,b) max(a,b);
                out = zeros(12,12,length(SV))-2;
            elseif strcmp(upper(typeflag),"MIN")
                fcn = @(a,b) min(a,b);
                out = 10 + zeros(12,12,length(SV));
            elseif strcmp(upper(typeflag(1:3)),"BAD")
                if idx ~= -1
                    error('Bad updates must be idx= -1')
                end
                badflag = true;
                fcn = @(a,b) a+b;
                out = zeros(12,12,length(SV));
            else
                error('Invalid typeflag')
            end
            
            
            W = waitbar(0,'Loading Capacitor Values');
            numb = 0;
            for k = 1:length(d)
                waitbar(k/length(d),W);
                matinfo = load([network_project_superclass2.ExperimentFolder,EG.ADDSTR,d(k).name],'experiment');
                s = find(SV==sortvar(k),1);
                if badflag
                    [a,b,c]= matinfo.experiment.problemUpdates();
                    
                    switch upper(typeflag)
                        case "BAD"
                            caps = a;
                        case "BADUP"
                            caps = b;
                        case "BADDOWN"
                            caps = c;
                    end
                    
                else
                    caps = matinfo.experiment.capacitors(idx);
                end
                if badflag || nanmax(caps(:)) ~= nanmin(caps(:))
                    out(:,:,s) = fcn(out(:,:,s) , caps);
                    numb = numb+1;
                end
            end
            close(W);
            if divon
                out = out/numb;
                
            else
                out(isnan(matinfo.experiment.capacitors(idx))) = nan;
            end
            if nargin>=6 && ~NOFIG
                figure(2209281)
                clf
                
                
                for s = 1:length(SV)
                    subplot(2,ceil(length(SV)/2),s)
                    
                    imagesc(out(:,:,s));
                    caxis([0,max(1,max(out(:)))])
                    colorbar
                    title(num2str(SV(s)))
                end
            end
            
        end
        
        
        % Finds units for a given quantity
        function unit = findUnit(EXP,tag)
            
            idx = find(strcmp(EXP.TAGS,tag));
            if isempty(idx) || isempty(char(EXP.UNITS(idx)))
                unit = '';
            else
                unit = [' (',char(EXP.UNITS(idx)),')'];
            end
        end
        
        
        % Finds divisor for a given quantity
        function mult = findMult(EG,tag)
            
            idx = find(strcmp(EG.TAGS,tag));
            if isempty(idx)
                mult = 1;
            else
                mult = EG.MULTS(idx);
            end
            
        end
        % Finds shift for a given quantity
        function shift = findShift(EG,tag)
            
            idx = find(strcmp(EG.TAGS,tag));
            if isempty(idx)
                shift = 0;
            else
                shift = EG.SHIFTS(idx);
            end
            
        end
        
        function [EXP,num] = loadBest(EG,tag,typeflag,want)
            
            if nargin<2
                tag = 'err1';
            end
            if nargin<3 || isempty(typeflag)
                typeflag = 'min';
            end
            val = EG.compoundValue(tag);
            
            
            % filter values if requested
            if nargin >=4 && ~isempty(want)
                if isstruct(want)
                    
                    want = EG.getWant(want);
                end
                val(~want) = nan;
                
            end
            
            
            switch upper(typeflag)
                case 'MAX'
                    [~,num] = nanmax(val);
                case 'MIN'
                    [~,num] = nanmin(val);
                otherwise
                    error('invalid typeflag')
            end
            
            EXP = EG.loadExperiment(num);
            
        end % end of methods section
        
        
        function idxs = findEquivalent(EG,idx,want,varargin)
            
            if length(idx)~=1 || round(idx)~=idx || idx<1
                error('idx must be a positive scalar index')
            end
            if isempty(varargin)
                error('Must stipulate fields to match')
            end
            
            if isempty(want)
                want = true(size(EG.ERR1));
            end
            
            for k = 1:length(varargin)
                want = and(want,EG.(varargin{k})(idx)==EG.(varargin{k}));
            end
            idxs = find(want);
            
            idxs = idxs(idxs~= idx); % remove input index
            
        end
    end
end



