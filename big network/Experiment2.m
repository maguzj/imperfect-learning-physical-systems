classdef Experiment2 < network_project_superclass2
    % Class instance for a unique physical coupled learning network.
    
    %% Properties
    properties (Constant)
        
        ObjName = 'experiment';
        SaveFolder = network_project_superclass2.ExperimentFolder;
    end
    
    properties (SetAccess = immutable) %
        
        Name       % string, name of experiment
        % Data considerations:
        
        num_free_inputs = 8;
        
        % Scalars
        SOR % number of sources
        TAR  % number of targets
        EPO  % number of epochs
        TRA  % training set size
        TST  % test set size
        ETA; % nudge amp*129
        ALF = 100;  % learning time (in us)
        REP % measurement repetitions
        MES % number of measurements that will be taken
        NOR = 0 % if true, ALF is normalized by # (incorrectly classified) datapoints
        
        TNO = 0; % std of noise added to goal values (*1/1000)
        SNO = 0; % std of noise added to source values (*1/1000)

        GRU = 1; % number of training groups
        
        ANA = 1; % do analog measurements for caps instead of serialized digital
        
        % learning rate
        DUP = 1 % number of times a learning step is engaged (on and off) per training step
        
        % Conductance-related
        VMN = 18 % minimum voltage value (Vmin ~ 8*VMN/129)
        VIT = 45 % initialization voltage (Vinit ~ 8*VIT/129)
        % For initialization type 2: V_init ~ 8*VIT/129-.4V

        RES = -2000;%old:10^5 % microsec of edge reset performed at the start 
        % 1 means no reset, negative is new reset
        
        % clamping
        DEL = 0 % if true, targets are paired to determine outputs (only has effect for class basis/UPD,UPT)
        ANT = 0 % using 'anti-clamped' configuration if > 0
        EOO = 0; % ETA Over One means using the amplification clamping setup.
        TFB = 0; %Target FeedBack -- when DEL>0, adjust clamping iteratively to match desired.
        % classification-based
        CLA = 1 % number of classes (1 ignores classes)
        HOT = 0 % "one-hot" class representations if >0. "on" values are 0.5+ONEHOT/1000. Other values are 0.5-ONEHOT/1000.
        UPD = 1 % KEEP ON! (makes hinge loss) if true and a classification task, update only when incorrectly classified.
        % UPD=2 -> ETA->0 for "no update", UPD=3 -> target switched off for
        % no update. UPD=4:   ALF = heaviside(ALF*buffer*UPT/1000)

        UPT = 3 % (BUF*UPT/1000) is scale for no-measurement repeats of update/no update
        BUF = 0 % how far over the classification line does a datapoint need to be for selective updating purposes (dist is BUF/1000)
        
        AML = 1000; % adaptive learning rate multiplier per measurement (/1000)
        MNA = 100; % min alpha: all updates with UPD==4,5 will be multiples of this
        % don't bother
        REG = 0 % number of times (*1000) that edges are flickered per train step to drain caps (even if no real learning that step).
        BIO = 0 % use transition probability matrix to randomly cycle between train datapoints
        DAT = 0 % how many MES points arduino can go through without waiting for MATLAB to catch up. 0 means no waiting.
        DIF = 0 % 'diffuses' training value randomly with step size DIF/1000 (if DIF>0)
        SPR = 1 %  rate class bases spread from mean value (multiplied by 1+SPR/1000 times themselves, - SPR/1000*mean)
        VEG = 4 % number of vertical edges in each column (4 is periodic)
        HEG = 4 % number of horizontal edges in each row (4 is periodic)
        LOK = 1 % locking each step (vs continuous learning)
        BTH = 1 % if false, train and test sets are not measured in batches on MES steps, only current train datapoint is
        KIL = 0 % (if >0) after this many measurement rounds with perfect accuracy, stop classification tasks short.
        PARAMNAMES =["EPO", "TRA", "TST", "ETA", "ALF", "SOR", "TAR",...
            "REP","MES","RES","VMN","CLA","SPR","VIT","VEG","HEG","DUP",...
            "HOT","DIF","UPD","UPT","DEL","BUF","DAT","ANT","LOK","BTH",...
            "KIL","BIO","REG","GRU","ANA","TNO","SNO","TFB","NOR","AML","MNA"];
        
        
        
        % Arrays
        TRAIN % training set (SOR+TAR)x(TRA)
        SENDTRAIN % training set recast to account for output errors
        TEST % test set (SOR+TAR)x(TST)
        SENDTEST % test set recast to account for output errors
        DOTEST % indicees of testing schedule (1)x(MES)
        TARGETLOCS % integer locations of target nodes (1)x(TAR)
        % (note that row is first 2 digits of binary, col is last 2)
        % ORDER % order for training 1x(EPOxTRA) ## not needed anymore!
        SOURCELOCS
        ETAS % eta values at each measurement step (if 0, not used)
        TRAINCLASSES % classes of each datapoint (0s if irrelevant, indicated by CLA = 1;
        TESTCLASSES % classes of each datapoint (0s if irrelevant, indicated by CLA = 1;
        ALFS % ALF values at each measurement step (if 0, not used)
        DUPS % DUP values at each measurement step (if 0, not used)
        TRAINIDX % indicees of training set to use (<0 means cycle duplicating -TRAINIDX times, 0 means random)
        WAITS % microseconds of delay after each training step
        TRAINWAIT % microseconds of delay after each training datapoint (whenever it is selected)
        TARGFLAG % are clamped targets on or off for this training datapoint (default on)
        LEARNFLAG % are we doing learning for this training datapoint (default on) (only works with LOK=1)
        MEASURECAPS % are we measuring caps and node during this training datapoint (1,TRA)
        LEMURLEARN % these training datapoints learn if the previous one learned.
        TRAINGROUP % if we are switching between groups for training, indicate with 111,22222, etc
        TMAT % (if BIO=1, TRAINIDX=0) relative probabilities to transition between train set datapoints 
        % ordered to have large, small, large, small sizes to allow arduino to avoid getting bogged down before the next recieve check.
        DATANAMES = ["SENDTRAIN","SOURCELOCS","SENDTEST","TARGETLOCS","DOTEST","TRAINCLASSES","ETAS","TESTCLASSES","TRAINIDX","ALFS","WAITS","TRAINWAIT","TARGFLAG","DUPS","LEARNFLAG","TMAT","TRAINGROUP","MEASURECAPS","LEMURLEARN"]; %,"ORDER"];
        
        % Physical considerations:
        
        % Scalars of in-network values
        RCHARGE=1000; % charging resistors in Ohms
        % imag value on RDRAIN means connect to +8
        RDRAIN=Inf; % resistance from Z to ground always drainage resistors in Ohms
        CCHARGE= 2.2*10^-6;%.47*10^(-6); % charging capacitors in Farads
        % imag value on ROFF means connected to +8V
        %... after 1/20 it means connected to VMN
        ROFF=Inf ;% resistance from Z to ground when learning is off
        CEDGE = 0; %capacitance (F) across each edge of the network
        CNODE = 0; %capacitance (F) from each non-source node to ground.
        
        % Arrays
        NSIZE =[4,4] % network size in nodes (2 element array)
        SLOC % source locations (SORx2)
        TLOC % target locations (TARx2)
        BROKENLOC % Location of "Broken" Edge [nodeloc, 0 (horiz) or 1 (vert)]
        
        
        DIODES % list of diodes (node A to node B)
        
        RFREE_LOW = 10^4;
        RFREE_HIGH = 10^5;
        RCLAMP_LOW = 10^4;
        RCLAMP_HIGH = 10^5;
        
        SOURCERES = 0;
        TARGETRES = Inf;
        TARGETC = 0;% 2.2*10^-6
        
        TLNODE = [0,0] % node we start with
        NSIZE2 = [4,4] % just the edges we care about (disconnect the rest)
        ISPERIODIC = [1, 1];
        
        
        % values that will prompt a warning if changed
        % from their default values
        NOCHANGES = ["RCHARGE","RDRAIN","CCHARGE","ROFF"...
            ,"SOURCERES","TARGETRES","TARGETC","CEDGE","CNODE"]
        
        OTHERNEEDED = ["NSIZE","SLOC","TLOC"];
    end
    
    properties
        AMPSETUP = 'One 10x per network';

        Notes % string, whatever.
        
        PREVEXP = ''; % name of previous experiment (if any)
        
        TestMeasurements % (TAR)x(TST)x(MES)
        TrainMeasurements % (TAR)x(TRA)x(MES)
        TestMeasurementsClamped% (TAR)x(TST)x(MES)
        TrainMeasurementsClamped% (TAR)x(TRA)x(MES)
        LearnTimes % 1x(MES)
        AbsoluteTimes % 1x(MES)
        
        TestMSE % TARx(MES)
        TrainMSE % TARx(MES)
        TrainingMSE % TARx(MES) (all three are TAR/2 if DEL==1)
        
        TestError % TARxTSTx(MES)
        TrainError % TARxTRAx(MES)
        
        
        TestMSE2 % TARx(MES) %calc directly from measurements...
        TrainMSE2 % TARx(MES) % not from supposed input values
        
        
        TestMeanMSE % TARx(MES) error calc from mean of test set
        TrainMeanMSE % TARx(MES) error calc from mean of training set
        
        TestMeanMeanMSE % TARx(MES) error calc: mean output minus mean answers
        TrainMeanMeanMSE % TARx(MES) error calc:  mean output minus mean answers
        
        TestConfusion % CLAxCLAx(MES) (real class, output class)
        TrainConfusion % CLAxCLAx(MES)
        
        HorizontalCapacitors % (NSIZE(1))x(NSIZE(2))x(MES)
        VerticalCapacitors   % (NSIZE(1))x(NSIZE(2))x(MES) (prior to update for each DOTEST train step)
        
        HorizontalCapacitorsTEST = []  % (NSIZE(1))x(NSIZE(2))x(MES)x(TST) (used only in manual tests)
        VerticalCapacitorsTEST = []
        
        % Last training step used to trin the network for each MES step
        
        TrainFreeState % (NSIZE(1))x(NSIZE(2))x(MES)
        TrainClampedState   % (NSIZE(1))x(NSIZE(2))x(MES) (prior to update for each DOTEST train step)
        
        % TEST set measured in its entirety every MES step
        
        TestFreeState % (NSIZE(1))x(NSIZE(2))x(MES) x (TST)
        TestClampedState   % (NSIZE(1))x(NSIZE(2))x(MES) x (TST) (prior to update for each DOTEST train step)
        
        
        
        trainStep % vector of each training step, which train datapoint was used
        
        ClassBasis   % TARxCLAxMES (prior to update for each DOTEST train step)
        
        % values for power/current calculation
        dG % gate voltage drop
        dS% source-drain voltage drop
        C% source-drain current
        
        
        
        isout % function to determine if capacitors are at max/min values
        
        % to convert from measurements/parameters to voltages
        % Ktop = (1/100)^-1;
   % Kbottom = 1000;
   % GATEMULT = 5*(Ktop+Kbottom)/Kbottom;
   
    GATEMULT = 9.97;
      %  GATEMULT = 5.5;
        NODEMULT = 4.985/11;
        VITMULT = 2.25*8/129;
        VMNMULT = 8/129;
        VITSHIFT = -1.25;
        VMNSHIFT = -.37;
    end
    
    
    %% Methods
    methods
        
        % returns a brand spankin new experiment object with the same
        % properties as this one.
        function EXP2 = duplicate(EXP,NEWSCALARS,NEWARRAYS)
            SCALARS = struct();
            ARRAYS = struct();
            SNAMES = [EXP.PARAMNAMES, EXP.NOCHANGES];
            DNAMES = [EXP.DATANAMES,"TRAIN","TEST",EXP.OTHERNEEDED];
            %SHOULD LOOK MORE CAREFULLY FOR SMALL STUFF
            
            % copy old parameters
            for k = 1:length(SNAMES)
                SCALARS.(SNAMES(k)) = EXP.(SNAMES(k));
            end
            for k = 1:length(DNAMES)
                ARRAYS.(DNAMES(k)) = EXP.(DNAMES(k));
            end
            
            if nargin>=2 && ~isempty(NEWSCALARS)
                fn = fieldnames(NEWSCALARS);
                for k=1:numel(fn)
                    SCALARS.(fn{k}) = NEWSCALARS.(fn{k});
                end
            end
            
            if nargin>=3 && ~isempty(NEWARRAYS)
                fn = fieldnames(NEWARRAYS);
                for k=1:numel(fn)
                    ARRAYS.(fn{k}) = NEWARRAYS.(fn{k});
                end
            end
            
            
            
            EXP2 = Experiment2([EXP.Name,'_COPY'],SCALARS,ARRAYS);
            
        end
        
        % returns the proper dimensions of a given property, expressed as
        % scalar properties of the object.
        function out = getDimensions(EXP,propname)
            
            if sum(strcmpi(propname,{'TRAIN','SENDTRAIN'}))
                out = {'SOR+TAR', 'TRA'};
            elseif sum(strcmpi(propname,{'TEST','SENDTEST'}))
                out = {'SOR+TAR', 'TST'};
            elseif sum(strcmpi(propname,{'TRAINCLASSES','TARGFLAG','TRAINWAIT','LEARNFLAG','TRAINGROUP','MEASURECAPS','LEMURLEARN'}))
                out = {1, 'TRA'};
            elseif sum(strcmpi(propname,{'TESTCLASSES'}))
                out = {1, 'TST'};
            elseif sum(strcmpi(propname,{'WAITS','DOTEST','ALFS','ETAS','TRAINIDX','LearnTimes','AbsoluteTimes','DUPS'}))
                out = {1, 'MES'};
            elseif sum(strcmpi(propname,{'TARGETLOCS'}))
                out = {1, 'TAR'};
            elseif sum(strcmpi(propname,{'SOURCELOCS'}))
                out = {1,'SOR'};
            elseif sum(strcmpi(propname,{'TLOC'}))
                out = {'TAR', 2};
            elseif sum(strcmpi(propname,{'SLOC'}))
                out = {'SOR',2};
            elseif sum(strcmpi(propname,{'TestMeasurements'}))
                out = {'TAR', 'TST','MES'};
            elseif sum(strcmpi(propname,{'TrainMeasurements'}))
                out = {'TAR', 'TRA','MES'};
            elseif sum(strcmpi(propname,{'trainStep'}))
                out = {1,'EPO*TRA'};
            elseif sum(strcmpi(propname,{'TestMSE','TrainMSE','TestMSE2','TrainMSE2','TestMeanMSE','TrainMeanMSE','TestMeanMeanMSE','TrainMeanMeanMSE'}))
                out = {'TAR','MES'};
            elseif sum(strcmpi(propname,{'TestConfusion','TrainConfusion'}))
                out = {'CLA (real)','CLA (output)', 'MES'};
            elseif sum(strcmpi(propname,{'HorizontalCapacitors','VerticalCapacitors','TrainFreeState','TrainClampedState','TestFreeState','TestClampedState'}))
                out = {4,4,'MES'};
            elseif sum(strcmpi(propname,{'ClassBasis'}))
                out = {'TAR', 'CLA','MES'};
            elseif sum(strcmpi(propname,{'TMAT'}))
                out = {'TRA','TRA'};
            else
                out = 'Property not found';
            end
            
        end
        
        % Create an instance of Experiment2. All inputs required.
        function obj = Experiment2(Name,SCALARS,ARRAYS)
            
            obj@network_project_superclass2();
            obj.Name = Name;
            obj.FullPathSaveName = obj.fullFilename();
            
            if nargin<3
                error('Not enough inputs: need NAME and objects SCALARS & ARRAYS.')
            end
            
            % Check all scalars are scalars > 0, add to obj.
            fn = fieldnames(SCALARS);
            for k=1:numel(fn)
                if( isnumeric(SCALARS.(fn{k})) && ...
                        numel(SCALARS.(fn{k}))==1 ...
                        && (SCALARS.(fn{k})>0 || imag(SCALARS.(fn{k}))>0))
                    %disp(['obj.',fn{k},'=SCALARS.',fn{k},';'])
                    % store values
                    
                    if sum(strcmp(obj.NOCHANGES,fn{k})) && ~isequal(obj.(fn{k}),SCALARS.(fn{k}))
                        disp(['########## CHANGING ',fn{k},' #######'])
                    end
                    obj.(fn{k})=SCALARS.(fn{k});
                    
                elseif strcmp(fn{k},'TST') && SCALARS.(fn{k})==0
                    disp('########## TEST SET SIZE = 0 #######')
                    obj.(fn{k})=SCALARS.(fn{k}) ;
                elseif SCALARS.(fn{k})==0  && sum(strcmp(fn{k},...
                        {'HOT','DIF','DEL','UPD','LOK','BTH','BUF','DAT','BIO','REG',...
                        'SOURCERES','TARGETRES','TARGETC','CEDGE','ANT','UPT','EOO',...
                        'ANA','SNO','TNO','TFB','NOR','CNODE','CEDGE'}))% one-hot flag can be off
                    obj.(fn{k})=SCALARS.(fn{k}) ;
                elseif sum(strcmp(fn{k}, {'UPT','RES'})) && SCALARS.(fn{k})<-1 % UPT/RES can be negative
                    obj.(fn{k})=SCALARS.(fn{k}) ;
                else
                    error(['SCALARS.',fn{k},' is not a scalar > 0.'])
                end
            end
            
            if obj.ETA >129
                error('ETA must be <= 129')
            end
            
            if obj.EPO ~= round(obj.EPO)
                error('EPO must be a whole number')
            end
            
            if obj.DEL>0 && (rem(obj.TAR,2)~=0)
                error('Cannot do differential outputs with odd target count.')
            end
            
            % Check all arrays are numeric arrays, add to obj.
            fn = fieldnames(ARRAYS);
            for k=1:numel(fn)
                if( isnumeric(ARRAYS.(fn{k})))
                    %disp(['obj.',fn{k},'=SCALARS.',fn{k},';'])
                    % store arrays
                    obj.(fn{k})=ARRAYS.(fn{k});
                elseif (strcmp(fn{k},'TRAINCLASSES') || strcmp(fn{k},'TESTCLASSES'))...
                        && islogical(ARRAYS.(fn{k}))
                    
                    obj.(fn{k})=double(ARRAYS.(fn{k}));
                    
                else
                    error(['ARRAYS.',fn{k},' is not a numeric array.'])
                end
            end
            
            % Individual checks on arrays
            
            % Training set
            if (~isequal(size(obj.TRAIN) , [obj.SOR+obj.TAR,obj.TRA]))
                error(['ARRAYS.TRAIN must be size ',...
                    num2str(obj.SOR+obj.TAR),'x',...
                    num2str(obj.TRA),'.'])
            end
            
            if ~isfield(ARRAYS,'SENDTRAIN')
                obj.SENDTRAIN = obj.TRAIN;

            else
                
                % Sending Training set
                if (~isequal(size(obj.SENDTRAIN) , [obj.SOR+obj.TAR,obj.TRA]))
                    error(['ARRAYS.SENDTRAIN must be size ',...
                        num2str(obj.SOR+obj.TAR),'x',...
                        num2str(obj.TRA),'.'])
                end
  
            end
            
            
            % Test set
            if ~(obj.TST==0 && numel(obj.TEST)==0) && (~isequal( size(obj.TEST) , [obj.SOR+obj.TAR,obj.TST]))
                error(['ARRAYS.TEST must be size ',...
                    num2str(obj.SOR+obj.TAR),'x',...
                    num2str(obj.TST),'.'])
            end
            
            if ~isfield(ARRAYS,'SENDTEST')
                obj.SENDTEST = obj.TEST;

            else
                % Test set
                if (~isequal( size(obj.SENDTEST) , [obj.SOR+obj.TAR,obj.TST])) &&...
                        ~(obj.TST==0 && numel(obj.SENDTEST)==0)
                    error(['ARRAYS.SENDTEST must be size ',...
                        num2str(obj.SOR+obj.TST),'x',...
                        num2str(obj.TST),'.'])
                end
            end
            
            % SENDTRAIN and SENDTEST are 255x higher to make into 8-bit
            if mean((obj.SENDTRAIN(:)+1)./(obj.TRAIN(:)+1)) < 10
                obj.SENDTRAIN = round(obj.SENDTRAIN*255);
            end
            
            if mean((obj.SENDTEST(:)+1)./(obj.TEST(:)+1)) < 10
                obj.SENDTEST = round(obj.SENDTEST*255);
            end
            
          
            % Testing Schedule
            if (~isequal(size(obj.DOTEST,1) , 1))
                error('ARRAYS.DOTEST must have 1 row.')
            end
            
            
            if ~isfield(ARRAYS,'ETAS')
                obj.ETAS = obj.DOTEST*0; % Zero means don't change
            else
                % Test set
                if (~isequal( size(obj.DOTEST) , size(ARRAYS.ETAS)))
                    error('ARRAYS.ETAS must be same size as ARRAYS.DOTEST.')
                end
                
            end
            
            if ~isfield(ARRAYS,'ALFS')
                obj.ALFS = obj.DOTEST*0; % Zero means look at ALF
            else
                % Test set
                if (~isequal( size(obj.DOTEST) , size(ARRAYS.ALFS)))
                    error('ARRAYS.ALFS must be same size as ARRAYS.DOTEST.')
                end
                
            end
            
            if ~isfield(ARRAYS,'DUPS')
                obj.DUPS = obj.DOTEST*0; % Zero means look at ALF
            else
                % Test set
                if (~isequal( size(obj.DOTEST) , size(ARRAYS.DUPS)))
                    error('ARRAYS.DUPS must be same size as ARRAYS.DOTEST.')
                end
                
            end
            
            if ~isfield(ARRAYS,'WAITS')
                obj.WAITS = obj.DOTEST*0; % Zero means no waiting
            else
                % Test set
                if (~isequal( size(obj.DOTEST) , size(ARRAYS.WAITS)))
                    error('ARRAYS.WAITS must be same size as ARRAYS.DOTEST.')
                end
                
            end
            
            
            if ~isfield(ARRAYS,'TARGFLAG')
                obj.TARGFLAG = zeros(1,obj.TRA)+3; % 3 means both do clamp
            else
                if (~isequal( [1,obj.TRA] , size(ARRAYS.TARGFLAG)))
                    error('Size of ARRAYS.TARGFLAG must be [1,TRA].')
                end
                
            end
             if ~isfield(ARRAYS,'LEARNFLAG')
                obj.LEARNFLAG = zeros(1,obj.TRA)+1; % 1 means do learn
            else
                if (~isequal( [1,obj.TRA] , size(ARRAYS.LEARNFLAG)))
                    error('Size of ARRAYS.LEARNFLAG must be [1,TRA].')
                end
                
             end
             
             
             if ~isfield(ARRAYS,'MEASURECAPS')
                 obj.MEASURECAPS = zeros(1,obj.TRA)+1; % all 1's means always measure
             else
                 if (~isequal( [1,obj.TRA] , size(ARRAYS.MEASURECAPS)))
                     error('Size of ARRAYS.MEASURECAPS must be [1,TRA].')
                 end
                 
             end
             
             if ~isfield(ARRAYS,'LEMURLEARN')
                 obj.LEMURLEARN = zeros(1,obj.TRA); % all 0's means don't lemur
             else
                 if (~isequal( [1,obj.TRA] , size(ARRAYS.LEMURLEARN)))
                     error('Size of ARRAYS.LEMURLEARN must be [1,TRA].')
                 end
                 
             end
             if sum(obj.LEMURLEARN>0) && obj.UPD~=1
                 error('LEMURLEARN>0 is only configured for UPD=1')
             end
                 
              
             if ~isfield(ARRAYS,'TRAINGROUP')
                obj.TRAINGROUP = zeros(1,obj.TRA); % all 0's means same groups
            else
                if (~isequal( [1,obj.TRA] , size(ARRAYS.TRAINGROUP)))
                    error('Size of ARRAYS.TRAINGROUP must be [1,TRA].')
                end
                
                if ~isequal(unique(obj.TRAINGROUP),(0:max(obj.TRAINGROUP)))
                    error('TRAINGROUP must contain exactly and only values 0:N where N is the max group number');
                end
             end
            
             if ~isfield(ARRAYS,'TMAT')
                obj.TMAT = zeros(obj.TRA,obj.TRA)+1; % equal probabilities
            else
                if (~isequal( [obj.TRA,obj.TRA] , size(ARRAYS.TMAT)))
                    error('Size of ARRAYS.TMAT must be [TRA,TRA].')
                end
                
            end
            
            if ~isfield(ARRAYS,'TRAINIDX')
                % signal random train idx
                obj.TRAINIDX = obj.DOTEST*0; % zero means random
            else
                if (~isequal( size(obj.DOTEST) , size(ARRAYS.TRAINIDX)))
                    error('ARRAYS.TRAINIDX must be same size as ARRAYS.DOTEST.')
                end
                
            end
            
            % Network Size
            if (~isequal(size(obj.NSIZE) , [1,2]))
                error(['ARRAYS.NSIZE must be size ',...
                    num2str(1),'x',...
                    num2str(2),'.'])
            end
            
            % Source Locations
            if (~isequal(size(obj.SLOC) , [obj.SOR,2]))
                error(['ARRAYS.SLOC must be size ',...
                    num2str(obj.SOR),'x',...
                    num2str(2),'.'])
            end
            
            % Target Locations
            if (~isequal(size(obj.TLOC) , [obj.TAR,2]))
                error(['ARRAYS.TLOC must be size ',...
                    num2str(obj.TAR),'x',...
                    num2str(2),'.'])
            end
            
            for k = 1:obj.TAR
                obj.TARGETLOCS(k) = sum(obj.TLOC(k,:).*[4,1]);
                fprintf(['Target @(',num2str(obj.TLOC(k,1)),',',...
                    num2str(obj.TLOC(k,2)),') at int=',num2str(obj.TARGETLOCS(k)),'   '])
            end
            disp(' ');
            
            fprintf('Source')
            for k = 1:obj.SOR
                obj.SOURCELOCS(k) = sum(obj.SLOC(k,:).*[4,1]);
                fprintf([' @(',num2str(obj.SLOC(k,1)),',',...
                    num2str(obj.SLOC(k,2)),') at int=',num2str(obj.SOURCELOCS(k)),'  '])
            end
            disp(' ');
            
            
            if ~isfield(ARRAYS,'TRAINCLASSES')
                obj.TRAINCLASSES = zeros(1,obj.TRA);
            end
            
            
            if ~isfield(ARRAYS,'TRAINWAIT')
                obj.TRAINWAIT = zeros(1,obj.TRA); % Zero means no waiting
            else
                if (~isequal( [1,obj.TRA] , size(ARRAYS.TRAINWAIT)))
                    error('Size of ARRAYS.TRAINWAIT must be [1,TRA].')
                end
                
            end
            
            
            if ~isfield(ARRAYS,'TESTCLASSES')
                obj.TESTCLASSES = zeros(1,obj.TST);
            end
            
            if size(obj.TESTCLASSES,2) ~= size(obj.TEST,2)
                error('TESTCLASSES and TEST must have same # of columns')
            end
            if size(obj.TRAINCLASSES,2) ~= size(obj.TRAIN,2)
                error('TRAINCLASSES and TRAIN must have same # of columns')
            end
            
            if ~isfield(SCALARS,'CLA') && isfield(ARRAYS,'TRAINCLASSES')
                obj.CLA = max(1,length(unique(ARRAYS.TRAINCLASSES)));
            end
            
            if ~isfield(SCALARS,'CLA')
                obj.CLA = 1;
            end
            
            if ~isfield(SCALARS,'GRU') && isfield(ARRAYS,'TRAINGROUP')
                obj.GRU = max(1,length(unique(ARRAYS.TRAINGROUP)));
            elseif ~isfield(SCALARS,'GRU')
                obj.GRU = 1;
            end
           
            obj.trainStep = uint16(zeros(1,obj.EPO*obj.TRA));
            
            obj = load_power_helpers(obj);
            
        
            obj.save(); % saves object immediately
        end
        
        % Send paramters to the arduino
        function sendParameters(obj,arduinoObj)
            abortstring = 'Invalid Data Codon';
            for k = 1:length(obj.PARAMNAMES)
                codon = obj.PARAMNAMES{k};
                write(arduinoObj,[codon,';',num2str(obj.(codon)),';'],'char');
                %disp(['WROTE: ',codon,';',num2str(obj.(codon)),';'])
                pause(0.01);
            end
            pause(0.2);
            idx = 0;
            while arduinoObj.NumBytesAvailable
                msg = readMsg(arduinoObj,';');
                if strcmp(msg,'Aborting experiment.')
                    error('Experiment aborted, Arduino reset.')
                    
                else
                    if idx<5
                        fprintf([msg,' '])
                        idx = idx+1;
                    else
                        idx = 0;
                        disp(msg);
                    end
                end
            end
            disp(' ');
        end
        
        
        
        % Send data (train/test set, order, test schedule) to arduino
        function sendData(obj,arduinoObj)
            abortstring = 'Invalid Data Codon';
            
            chunk = 16;
            waitbars = 50;
            
            % send each data array to arduino
            for k = 1:length(obj.DATANAMES)
                fprintf([obj.DATANAMES{k},': '])
                for j = 1:(12-length(obj.DATANAMES{k}))
                    fprintf(' ');
                end
                %signal to arduino which array is coming
                write(arduinoObj,[obj.DATANAMES{k},';'],'char');
                %disp([obj.DATANAMES{k},';'])
                data_obj = obj.(obj.DATANAMES{k}); % MATLAB objects copy automatically
                
                if length(unique(data_obj)) == 1
                    % if only one value, dont waste time sending them all
                    write(arduinoObj,['SOLO;'],'char');
                    write(arduinoObj,[num2str(data_obj(1,1)),';'],'char');
                    fprintf('[00001] ');
                    
                else % if heterogeneous, send 'em
                    write(arduinoObj,['MULTI;'],'char');
                    tot = numel(data_obj);
                    give_prog = tot > chunk;
                    checknum = chunk;
                    wrote = 0;
                    fprintf(['[',zeroify(tot,5),'] ']);
                    
                    fprintf('[');
                    for n=1:size(data_obj,1)
                        
                        for m=1:size(data_obj,2)
                            %disp([num2str(data_obj(n,m)),';'])
                            write(arduinoObj,[num2str(data_obj(n,m)),';'],'char');
                            
                            num = ((n-1)*size(data_obj,2)+m);
                            if give_prog && num == checknum
                                newrote = round(waitbars*num/tot);
                                while wrote<newrote
                                    fprintf('|');
                                    wrote = wrote+1;
                                end
                                checknum = checknum + chunk;
                                pause(0.1);
                            end
                        end
                    end
                    while wrote<waitbars
                        fprintf('|');
                        wrote = wrote+1;
                    end
                    fprintf(']');
                end
                fprintf([' SENT -> '])
                pause(0.1);
                idx = 0;
                while true
                    if arduinoObj.NumBytesAvailable
                        msg = readMsg(arduinoObj,';');
                        if strcmp(msg,[obj.DATANAMES{k},' recieved!'])
                            disp('RECIEVED');
                            pause(0.2)
                            break
                        else
                            disp(['READ: ',msg])
                            if strcmp(msg(1:length(abortstring)),abortstring)
                                error('failed to send data correctly')
                            end
                        end
                    else
                        if idx>20
                            idx=0;
                            disp('%')
                        else
                            idx = idx+1;
                            fprintf('%')
                        end
                        pause(0.5);
                    end
                    
                end
            end
            
        end
        
        
        function [out,outClamp,outGoal,outIdx] = TrainingMeasurements(EXP)
            
            T = EXP.TLOC + 1; % shift for matlab indexing
            
            out = zeros(EXP.TAR,EXP.MES);
            outClamp = zeros(EXP.TAR,EXP.MES);
            
            for t = 1:EXP.TAR
                out(t,:) = squeeze(EXP.TrainFreeState(T(t,1),T(t,2),:));
                outClamp(t,:) = squeeze(EXP.TrainClampedState(T(t,1),T(t,2),:));
            end
            
            if nargout>=3 % find goals and id's
                outGoal = nan(EXP.TAR,EXP.MES);
                outIdx = nan(1,EXP.MES);
                for m = 1:EXP.MES
                    idx = EXP.TRAINIDX(m);
                    if idx >0
                        outGoal(:,m) = EXP.TRAIN(EXP.SOR+1:end,idx);
                        outIdx(:,m) = idx;
                    elseif idx < 0
                        idx0 = EXP.DOTEST(m)-1;
                        modular = EXP.TRA*-idx;
                        realidx = floor(rem(idx0,modular)/-idx)+1;
                        outGoal(:,m) = EXP.TRAIN(EXP.SOR+1:end,realidx);
                        outIdx(:,m) = realidx;
                    elseif idx == 0 && EXP.TRA == 1
                        outGoal(:,m) = EXP.TRAIN(EXP.SOR+1:end,1);
                        outIdx(:,m) = 1;
                    end
                end
            end
            
        end
        
        function obj = CalcError(obj)
            
            
            % calculate trainING measurement (just from measured datapoints
            % used during the training process, no batch testing).
            [out,~,outGoal,~] = obj.TrainingMeasurements();
            if obj.DEL > 0
                out =out(2:2:end,:)-out(1:2:end,:);
                outGoal =outGoal(2:2:end,:)-outGoal(1:2:end,:);
            end
            
            obj.TrainingMSE = (outGoal-out).^2;
            
            
            dotest = obj.TST>0;
            
            test_ans = obj.TEST((obj.SOR+1):end,:); %(TAR x TST)
            train_ans = obj.TRAIN((obj.SOR+1):end,:); %(TAR x TRA)
            
            test_meas = obj.TestMeasurements(1:obj.TAR,:,:);%(TAR x TST x (MES))
            train_meas = obj.TrainMeasurements(1:obj.TAR,:,:);%(TAR x TRA x (MES))
            %             test_meas2 = test_meas*nan;
            %
            %             T = obj.TLOC + 1; % shift for matlab indexing
            %
            %             for t = 1:size(T,1)
            %                 test_meas2(t,:,:) = permute(obj.TestFreeState(T(t,1),T(t,2),:,:),[1,4,3,2]);
            %             end
            %
            %             figure()
            %             plot(test_meas(:),test_meas2(:),'o')
            %
            num_meas = size(obj.TestMeasurements,3);
            
            obj.TestMSE = zeros(obj.TAR, num_meas);
            obj.TrainMSE = zeros(obj.TAR, num_meas);
            
            if obj.DEL>0 % if doing differential outputs
                train_ans0 = zeros(obj.TAR/2,obj.TRA);
                test_ans0 = zeros(obj.TAR/2,obj.TST);
                train_meas0 = zeros(obj.TAR/2,obj.TRA,obj.MES);
                test_meas0 = zeros(obj.TAR/2,obj.TST,obj.MES);
                
                for targ = 1:obj.TAR/2 %
                    train_ans0(targ,:) = train_ans((targ*2-1),:)-train_ans(targ*2,:);
                    train_meas0(targ,:,:) = train_meas((targ*2-1),:,:) - train_meas(targ*2,:,:);
                    
                    if dotest
                        test_ans0(targ,:) = test_ans((targ*2-1),:)-test_ans(targ*2,:);
                        test_meas0(targ,:,:) = test_meas((targ*2-1),:,:) - test_meas(targ*2,:,:);
                    end
                end
                train_ans = train_ans0;
                test_ans = test_ans0;
                train_meas = train_meas0;
                test_meas = test_meas0;
                
                obj.TestMSE = zeros(obj.TAR/2, num_meas);
                obj.TrainMSE = zeros(obj.TAR/2, num_meas);
                
            end
            
            
            
            
            for t = 1:num_meas
                obj.TrainMSE(:,t) = mean(abs(squeeze(train_ans-train_meas(:,:,t)).^2),2);
                obj.TrainError(:,:,t) = train_ans-train_meas(:,:,t); % TARxTRAx(MES)
                
                if dotest
                    obj.TestMSE(:,t) = mean(abs(squeeze(test_ans-test_meas(:,:,t)).^2),2);
                    obj.TestError(:,:,t) = test_ans-test_meas(:,:,t); % TARxTSTx(MES)
                else
                    obj.TestMSE(:,t) = 0;
                end
            end
            
            %% error from mean training/test set
            if obj.DEL == 0
                test_meanmean = mean(test_ans,2);
                train_meanmean = mean(train_ans,2);
                test_mean = repmat(test_meanmean,[1,obj.TST]);
                train_mean = repmat(train_meanmean,[1,obj.TRA]);
                
                
                
                obj.TestMeanMSE = zeros(obj.TAR, num_meas);
                obj.TrainMeanMSE = zeros(obj.TAR, num_meas);
                
                obj.TestMeanMeanMSE = zeros(obj.TAR,num_meas);
                obj.TrainMeanMeanMSE = zeros(obj.TAR,num_meas);
                
                for t = 1:num_meas
                    obj.TrainMeanMSE(:,t) = mean(abs(squeeze(train_mean-train_meas(:,:,t)).^2),2);
                    obj.TrainMeanMeanMSE(:,t) = abs(squeeze(train_meanmean-mean(train_meas(:,:,t),2)).^2);
                    
                    if dotest
                        obj.TestMeanMSE(:,t) = mean(abs(squeeze(test_mean-test_meas(:,:,t)).^2),2);
                        obj.TestMeanMeanMSE(:,t) = abs(squeeze(test_meanmean-mean(test_meas(:,:,t),2)).^2);
                    else
                        obj.TestMeanMSE(:,t) = 0;
                        obj.TestMeanMeanMSE(:,t) = 0;
                    end
                end
                
            else
                disp('DEL > 0 and meanMSE combo not coded yet.')
            end
            
            % second way to calculate error
            T = obj.TLOC + 1; % shift for matlab indexing
            
            if sum(obj.ETAS) == 0
                etavec = obj.ETAS + obj.ETA/129;
            else
                etavec = obj.ETAS/129;
            end
            
            trainerrvec = 0;
            testerrvec= 0;
            for t = 1:obj.TAR*(1/(obj.DEL+1))
                
                for tst = 1:obj.TST
                    if obj.DEL
                        x = squeeze(obj.TestFreeState(T(2*t-1,1),T(2*t-1,2),:,tst))-...
                            squeeze(obj.TestFreeState(T(2*t,1),T(2*t,2),:,tst));
                        
                        y = squeeze(obj.TestClampedState(T(2*t-1,1),T(2*t-1,2),:,tst))-...
                            squeeze(obj.TestClampedState(T(2*t,1),T(2*t,2),:,tst));
                    else
                        x = squeeze(obj.TestFreeState(T(t,1),T(t,2),:,tst));
                        y = squeeze(obj.TestClampedState(T(t,1),T(t,2),:,tst));
                    end
                    testerrvec = testerrvec + (1/obj.TST)*((x-y)./(etavec')).^2;

                end
                obj.TestMSE2(t,:) = testerrvec;   % TARx(MES)
                
            end
            
            
            %
            
        end
        
        
        function ShowSourceDiscrepancies(obj,meas_step)
            disp('For non-allosteric tasks, discrepancies may be train step mismatch!')
            if nargin<2
                meas_step = length(obj.LearnTimes);
            end
            lastState = obj.TEST(:,1);
            EnforcedClampedState = nan(obj.NSIZE);
            S = size(obj.SLOC,1);
            for s = 1:S
                EnforcedClampedState(obj.SLOC(s,1)+1,obj.SLOC(s,2)+1) = lastState(s);
            end
            
            for t = 1:size(obj.TLOC,1)
                EnforcedClampedState(obj.TLOC(t,1)+1,obj.TLOC(t,2)+1) = lastState(S+t);
            end
            
            
            
            figure(2206301)
            clf
            
            titles = {'Enforced','Measured','DIFF'};
            caxes = {[0,1],[0,1],[-.1,.1]};
            values = {EnforcedClampedState,obj.TrainClampedState(:,:,meas_step),(EnforcedClampedState-obj.TrainClampedState(:,:,meas_step))};
            for k = 1:3
                subplot(1,3,k)
                imagesc(values{k})
                hold on
                plot(obj.SLOC(:,2)+1,obj.SLOC(:,1)+1,'ks','MarkerSize',15,'LineWidth',3)
                plot(obj.TLOC(:,2)+1,obj.TLOC(:,1)+1,'ko','MarkerSize',15,'LineWidth',3)
                caxis(caxes{k})
                title(titles{k})
                
                set(gca,'XTick',1:4,'XTickLabels',0:3,'YTick',1:4,'YTickLabels',0:3)
            end
            
            colorbar
            
            
        end
        
        function [appliedVals,realVals,goalVals,realValsFree] = RealAndAppliedVals(EXP)
            
            
            realVals = zeros([(EXP.SOR+EXP.TAR),EXP.TST,EXP.MES]);
            realValsFree = zeros([EXP.SOR,EXP.TST,EXP.MES]);
            goalVals = realVals;
            LOCS = [EXP.SLOC; EXP.TLOC];
            for s = 1:EXP.SOR+EXP.TAR
                r = LOCS(s,1)+1;
                c = LOCS(s,2)+1;
                RV = EXP.TestClampedState(r,c,:,:);
                realVals(s,:,:) = squeeze(RV)';
                
                if s<=EXP.SOR
                    RVF = EXP.TestFreeState(r,c,:,:);
                    realValsFree(s,:,:) = squeeze(RVF)';
                end
                for t = 1:size(EXP.TEST,2)
                    classnum = EXP.TESTCLASSES(t)+1;
                    if EXP.CLA<=1 || s<=EXP.SOR
                        goalVals(s,t,:) = EXP.TEST(s,t);
                    else
                        goalVals(s,t,:) = EXP.ClassBasis(s-EXP.SOR,classnum,:);
                    end
                end
            end
            
            
            if EXP.DEL>0
                goalVals2 = goalVals([1:EXP.SOR, EXP.SOR + (2:2:EXP.TAR)],:,:);
                goalVals2(EXP.SOR+1:end,:,:) = goalVals2(EXP.SOR+1:end,:,:) - goalVals(EXP.SOR + (1:2:EXP.TAR),:,:);
                goalVals = goalVals2;
                
                realVals2 = realVals([1:EXP.SOR, EXP.SOR + (2:2:EXP.TAR)],:,:);
                realVals2(EXP.SOR+1:end,:,:) = realVals2(EXP.SOR+1:end,:,:) - realVals(EXP.SOR + (1:2:EXP.TAR),:,:);
                realVals = realVals2;
            end
            
            % targets are applied not at goal but at nudge
            appliedVals = goalVals;
            ETASvals = EXP.ETAS;
            if sum(ETASvals)==0
                ETASvals = ETASvals + EXP.ETA;
            end
            ETASvals = ETASvals/129;
            
            if EXP.DEL > 0
                for idx = 1:EXP.MES
                    for t0 = 1:(EXP.TAR/2)
                        
                        r = EXP.TLOC(2*t0,1)+1;
                        c = EXP.TLOC(2*t0,2)+1;
                        r1 = EXP.TLOC(2*t0-1,1)+1;
                        c1 = EXP.TLOC(2*t0-1,2)+1;
                        % measured free values for each test set data
                        FV = EXP.TestFreeState(r,c,idx,:)-EXP.TestFreeState(r1,c1,idx,:);
                        
                        GV = goalVals(EXP.SOR+t0,:,idx);
                        if EXP.ANT>0 % if anti-clamping, use new 'label'
                            GV = squeeze(FV)'*2-GV;
                        end
                        appliedVals(EXP.SOR+t0,:,idx) =...
                            squeeze(FV*(1-ETASvals(idx)))' + GV*ETASvals(idx);
                        
                    end
                end
                
                
            else
                for idx = 1:EXP.MES
                    for t = 1:EXP.TAR
                        
                        r = EXP.TLOC(t,1)+1;
                        c = EXP.TLOC(t,2)+1;
                        % measured free values for each test set data
                        FV = EXP.TestFreeState(r,c,idx,:);
                        GV = goalVals(EXP.SOR+t,:,idx);
                        if EXP.ANT>0 % if anti-clamping, use new 'label'
                            GV = squeeze(FV)'*2-GV;
                        end
                        appliedVals(EXP.SOR+t,:,idx) =...
                            squeeze(FV*(1-ETASvals(idx)))' + GV*ETASvals(idx);
                        
                    end
                end
            end
            
            realVals = realVals*EXP.NODEMULT;
            realValsFree = realValsFree*EXP.NODEMULT;
            appliedVals = appliedVals*EXP.NODEMULT;
            goalVals = goalVals*EXP.NODEMULT;
            
        end
        
        function [outmean,outmax] = discrepancySum(EXP,idx)
            
            if nargin>1
                for k = 1:length(idx)
                    if idx(k)<0
                        idx(k) = EXP.MES + idx(k);
                    end
                end
            else
                idx = 1:EXP.MES;
            end
            
            [appliedVals,realVals,~] = EXP.RealAndAppliedVals();
            appliedVals = appliedVals(:,:,idx);
            realVals = realVals(:,:,idx);
            
            outmean = sum(squeeze(mean((appliedVals-realVals).^2,2)),1);
            if nargout>1
                outmax = max(squeeze(max((appliedVals-realVals).^2,[],2)),[],1);
            end
        end
        
        function DiscrepancySummary(EXP)
            
            [appliedVals,realVals,~,realValsFree] = EXP.RealAndAppliedVals();
            
            %             if EXP.DEL>0
            %                 appliedVals2 = appliedVals([1:EXP.SOR, EXP.SOR + (2:2:EXP.TAR)],:,:);
            %                 appliedVals2(EXP.SOR+1:end,:,:) = appliedVals2(EXP.SOR+1:end,:,:) - appliedVals(EXP.SOR + (1:2:EXP.TAR),:,:);
            %                 appliedVals = appliedVals2;
            %
            %                 realVals2 = realVals([1:EXP.SOR, EXP.SOR + (2:2:EXP.TAR)],:,:);
            %                 realVals2(EXP.SOR+1:end,:,:) = realVals2(EXP.SOR+1:end,:,:) - realVals(EXP.SOR + (1:2:EXP.TAR),:,:);
            %                 realVals = realVals2;
            %
            %             end
            figure(2211182)
            clf
            
            timeVals = cumsum(EXP.LearnTimes)/1000; % in ms
            
            
            
            % SOR+TAR, TST, MES (dimensions)
            % after sum just SOR+TAR, MES
            discrepancySum = squeeze(mean((appliedVals-realVals).^2,2));
            discrepancySumFree = squeeze(mean((appliedVals(1:EXP.SOR,:,:)-realValsFree).^2,2));
            leg = {};
            for s = 1:size(appliedVals,1) %EXP.SOR+EXP.TAR (halve TAR if DEL>0)
                if s<= EXP.SOR
                    col = [1,0,0]*(s/EXP.SOR*0.5 + 0.5);
                    leg{end+1} = ['Source ',num2str(s-1),' Clamp'];
                    leg{end+1} = ['Source ',num2str(s-1),' Free'];
                    
                else
                    col = [0,0,1];
                    leg{end+1} = ['Target ',num2str(s-EXP.SOR-1),' Clamp'];
                end
                
                if std(EXP.TEST(s,:))>10^-10
                    str = '--';
                else
                    str = '-';
                end
                subplot(1,4,1)
                
                loglog(timeVals,discrepancySum(s,:),str,'LineWidth',2,'Color',col)
                hold on
                if s<=EXP.SOR
                    loglog(timeVals,discrepancySumFree(s,:),str,'LineWidth',1,'Color',col)
                end
                
                
                
                
            end
            xlabel('Learning Time (ms)')
            ylabel('Mean Discrepancy by Source, Target (V^2)')
            
            loglog(timeVals,sum(discrepancySum,1),'o','LineWidth',2,'Color','k')
            leg{end+1} = 'Sum';
            legend(leg)
            
            subplot(1,4,2)
            cols = @(x) x/size(appliedVals,3)*[1,0,1];
            t2 = round(size(appliedVals,3)/2);
            for t = [1,t2,size(appliedVals,3)]
                x = realVals(:,:,t);
                xF = realValsFree(:,:,t);
                y = appliedVals(:,:,t);
                yF = appliedVals(1:EXP.SOR,:,t);
                plot(y(:),x(:),'o','Color',cols(t)...
                    ,'LineWidth',2)
                hold on
                plot(yF(:),xF(:),'s','Color',cols(t)...
                    ,'LineWidth',2)
                hold on
                
            end
            plot([0,.6],[0,.6],'--')
            legend({'Initial Clamp','Initial Free',['MES=',num2str(t2),' Clamp'],['MES=',num2str(t2),' Free'],'Final Clamp','Final Free'})
            ylabel('Real Values (V)')
            xlabel('Applied Values (V)')
            
            networkDifference = squeeze(mean((realVals(1:EXP.SOR,:,:)-realValsFree).^2,2));
            
            subplot(1,4,3)
            for t = 1:EXP.MES
                
                plot(realVals(1:EXP.SOR,:,t),realValsFree(:,:,t),'o','Color',cols(t),'LineWidth',2)
                hold on
                
            end
            xlabel('Clamped Network Values (V)')
            ylabel('Free Network Values (V)')
            plot([0,.5],[0,.5],'k-')
            
            subplot(1,4,4)
            a = abs(EXP.TrainClampedState-EXP.TrainFreeState);
            b = abs(EXP.TestClampedState-EXP.TestFreeState);
            
            b = mean(b,4);
            b = mean(b,3);
            
            imagesc(log10(b));
            title('Test Free/Clamped Discrepancy')
            hold on
            for k = 1:EXP.SOR
                plot(EXP.SLOC(k,2)+1,EXP.SLOC(k,1)+1,'ko','LineWidth',4,'MarkerSize',15)
            end
            for k = 1:EXP.TAR
                plot(EXP.TLOC(k,2)+1,EXP.TLOC(k,1)+1,'k^','LineWidth',4,'MarkerSize',15)
            end
        end
        
        
        function PlotError(obj,flag)
            shapes = {'o','d','^','s','<','>'};
            
            if isempty(obj.TestMSE) || isempty(obj.TrainMSE)|| isempty(obj.TestMSE2)|| isempty(obj.TestMeanMSE)|| isempty(obj.TestMeanMeanMSE)
                obj = obj.CalcError();
            end
            
            time = cumsum(obj.LearnTimes)/1000; % in ms
            
            figure(2204201)
            clf
            errs = {obj.TrainMSE,obj.TestMSE,obj.TestMeanMSE,obj.TestMeanMSE};
            for errchoice = 0%:(obj.DEL==0)
               % subplot(2,1,errchoice+1)
                
                for k = 1:obj.TAR/(1+(obj.DEL>0))
                    
                    loglog(time,errs{errchoice*2+1}(k,:),shapes{k},'MarkerFaceColor',ones(1,3)*0.5,'Color',ones(1,3)*0.4);
                    hold on
                    loglog(time,errs{errchoice*2+2}(k,:),shapes{k},'MarkerFaceColor',ones(1,3)*0.1,'Color',ones(1,3)*0);
                end
                
                loglog(time,sum(errs{errchoice*2+1}(:,:)),'-','MarkerFaceColor',ones(1,3)*0.5,'Color',ones(1,3)*0.4,'LineWidth',2);
                hold on
                loglog(time,sum(errs{errchoice*2+2}(:,:)),'-','MarkerFaceColor',ones(1,3)*0.1,'Color',ones(1,3)*0,'LineWidth',2);
                
                xlabel('Learning Time (ms)')
                ylabel('Mean Sq Error')
                legend({'Train','Test'})
                xlim([0,time(end)*2-time(end-1)]);
            end
            
%             if nargin>=2 && flag==1
%                 figure(2212121)
%                 clf
%                 
%                 loglog(time,sum(obj.TrainMSE),'-','Color',ones(1,3)*0.4,'LineWidth',2);
%                 hold on
%                 loglog(time,sum(obj.TestMSE),'-','Color',ones(1,3)*0,'LineWidth',2);
%                 
%                 loglog(time,sum(obj.TrainMeanMSE),'--','Color',ones(1,3)*0.4,'LineWidth',2);
%                 hold on
%                 loglog(time,sum(obj.TestMeanMSE),'--','Color',ones(1,3)*0,'LineWidth',2);
%                 
%                 
%                 loglog(time,sum(obj.TrainMeanMeanMSE),':','Color',ones(1,3)*0.4,'LineWidth',2);
%                 hold on
%                 loglog(time,sum(obj.TestMeanMeanMSE),':','Color',ones(1,3)*0,'LineWidth',2);
%                 
%                 
%                 
%                 xlabel('Learning Time (ms)')
%                 ylabel('Mean Sq Error')
%                 legend({'Train','Test','Train (Mean)','Test (Mean)','Train (MM)','Test (MM)'})
%                 xlim([0,time(end)*2-time(end-1)]);
%             end
        end
        
        function PlotErrorvEpo(obj)
            shapes = {'o','d','^','s','<','>'};
            
            if isempty(obj.TestMSE) || isempty(obj.TrainMSE)
                obj.CalcError();
            end
            
            figure(2204201)
            clf
            
            xax = obj.DOTEST;
            
            for k = 1:obj.TAR
                loglog(xax,obj.TrainMSE(k,:),shapes{k},'MarkerFaceColor',ones(1,3)*0.5,'Color',ones(1,3)*0.4);
                hold on
                loglog(xax,obj.TestMSE(k,:),shapes{k},'MarkerFaceColor',ones(1,3)*0.1,'Color',ones(1,3)*0);
            end
            
            loglog(xax,sum(obj.TrainMSE(:,:)),'-','MarkerFaceColor',ones(1,3)*0.5,'Color',ones(1,3)*0.4,'LineWidth',2);
            hold on
            loglog(xax,sum(obj.TestMSE(:,:)),'-','MarkerFaceColor',ones(1,3)*0.1,'Color',ones(1,3)*0,'LineWidth',2);
            
            xlabel('Learning Step (EPO)')
            ylabel('Mean Sq Error')
            legend({'Train','Test'})
            xlim([0, 2*max(xax)]);
        end
        
        % display user instructions for wiring up this experiment
        function experimentInstructions(obj)
            disp(' ')
            disp('######################## Wiring Instructions ########################')
            
            for s = 1:obj.SOR
                disp(['   Output ',num2str(s-1),' (Source ',num2str(s),') goes to node (',num2str(obj.SLOC(s,1)),',',num2str(obj.SLOC(s,2)),'), BOTH (connect blue & yellow).'])
            end
            for s = 1:obj.TAR
                disp(['   Output ',num2str(obj.SOR+s-1),' (Target ',num2str(s),') goes to node (',num2str(obj.TLOC(s,1)),',',num2str(obj.TLOC(s,2)),'), CLAMPED (blue).'])
            end
            disp('-------------------- Do not connect more outputs. -------------------')
            
            for s = 1:obj.TAR
                disp(['   Input ',num2str(s),' (Target ',num2str(s),') goes to node (',num2str(obj.TLOC(s,1)),',',num2str(obj.TLOC(s,2)),'), FREE (yellow).'])
            end
            
            disp('--------- Additional inputs may be connected to the network. --------')
            disp(' ')
        end
        
        
        function PlotInsAndOuts(obj)
            colors = {'r','b','c','m','g','y','k'};
            time = cumsum(obj.LearnTimes)/1000; % in ms
            
            leg = {};
            figure(2204202)
            clf
            want = obj.DOTEST;
            for inp = 1:obj.TAR
                for t = 1:obj.TRA
                    semilogx(time,squeeze(obj.TrainMeasurements(inp,t,:)),[colors{inp},'-'])
                    hold on
                    leg{end+1} = ['Input #',num2str(inp),' Train Set #',num2str(t)];
                end
                for t = 1:obj.TST
                    plot(time,squeeze(obj.TestMeasurements(inp,t,:)),[colors{inp},'--'])
                    leg{end+1} = ['Input #',num2str(inp),' Test Set #',num2str(t)];
                end
            end
            
            
            xlabel('Learning Time (ms)')
            ylabel('Measured Values')
            legend(leg);
            xlim([0,time(end)*2-time(end-1)]);
        end
        
        
        function [H,V] = getTrainUpdate(obj,step,mult)
            if nargin<2
                step = obj.DOTEST(1);
            end
            if nargin<3
                mult = 1;
            end
            
            s = find(obj.DOTEST == step,1);
            if isempty(s)
                error('Requested update step was not measured.')
            end
            
            disp('Assuming periodicity')
            
            [H,V] = getUpdate(obj.TrainFreeState(:,:,s),obj.TrainClampedState(:,:,s));
            
            H = mult*10^-7*H*obj.ALF/obj.RCHARGE/obj.CCHARGE;
            V = mult*10^-7*V*obj.ALF/obj.RCHARGE/obj.CCHARGE;
        end
        
        function [H,V] = getAllUpdate(EXP,trainIdx,testIdx)
            
            
            if nargin<2 || isempty(trainIdx)
                trainIdx = 1:EXP.MES;
            end
            
            if nargin<3 || isempty(testIdx)
                testIdx = 1:EXP.TST;
            end
            
            trainIdx(trainIdx<0) = trainIdx(trainIdx<0)+EXP.MES+1;
            
            
            H = zeros(4,4,length(trainIdx),length(testIdx));
            V = zeros(4,4,length(trainIdx),length(testIdx));
            
            for k0 = 1:length(trainIdx)
                for j0 = 1:length(testIdx)
                    
                    [H0,V0] = getUpdate(EXP.TestFreeState(:,:,trainIdx(k0),testIdx(j0)),...
                        EXP.TestClampedState(:,:,trainIdx(k0),testIdx(j0)));
                    
                    H(:,:,k0,j0) = H0;
                    V(:,:,k0,j0) = V0;
                    
                end
            end
            
            
        end
        
        
        function misalignmentMovie(EXP,pausetime)
            
            if nargin<2
                pausecmd = @() pause;
            else
                pausecmd = @() pause(pausetime);
            end
            
            for k = 1:EXP.MES
                if k ==1
                    startval = EXP.plotMisalignments(k);
                else
                    EXP.plotMisalignments(k,startval);
                end
                pausecmd()
            end
        end
        
        function [out,badup,baddown] = problemUpdates(EXP)
            
            CP = EXP.capacitors(-1);
            UR = real(EXP.updateRule(-1));
            
            magcutoff = 10^-3;
            
            
            badup = and(UR>magcutoff,CP<3);
            baddown = and(UR<-magcutoff,CP>3);
            out = badup+baddown;
            
        end
        
        
        function startval = plotMisalignments(EXP,idx,startval)
            if EXP.TRA>1
                disp('Not allostery, this may fail.')
            end
            
            mksz = 10;
            
            if nargin<2
                idx = 1:EXP.MES;
            end
            
            UR = EXP.updateRule(idx);
            UR(isnan(UR)) = 0;
            UR = real(UR);
            CAP = EXP.capacitors([1,idx]);
            if nargin<3
                startval = nanmean(nanmean(CAP(:,:,1)));
            end
            CAP = CAP(:,:,2:end);
            
            
            figure(2210121);
            clf
            
            shps = '^sov';
            cols = 'cbmr';
            
            subplot(1,3,2)
            
            imagesc(UR(:,:,end));
            xL = [min(UR(:)),max(UR(:))];
            
            caxis([-1,1]*0.003)
            title('Update Rule')
            colormap(posnegColormap())
            colorbar;
            hold on
            
            subplot(1,3,3)
            CAP(isnan(CAP)) = startval;
            imagesc(CAP(:,:,end));
            caxis(startval+[-3,+3])
            
            title('Capactor Value')
            colormap(posnegColormap())
            colorbar
            hold on
            
            
            for r = 1:4
                for c = 1:4
                    
                    subplot(1,3,1)
                    
                    col = cols(c);
                    shp = shps(r);
                    % plot horiz caps
                    plot(squeeze(UR(r*3-2,c*3,:)),squeeze(CAP(r*3-2,c*3,:)),['-',shp],'Color',col,'MarkerFaceColor',col,'LineWidth',1.5,'MarkerSize',10)
                    hold on
                    % plot horiz caps
                    plot(squeeze(UR(r*3,c*3-2,:)),squeeze(CAP(r*3,c*3-2,:)),['-',shp],'Color','k','MarkerFaceColor',col,'LineWidth',1.5,'MarkerSize',10)
                    hold on
                    
                    for k = 2:3
                        subplot(1,3,k)
                        % horiz caps
                        plot(c*3-0.5,r*3-2,shp,...
                            'Color',col,'MarkerFaceColor',col,'MarkerSize',mksz,'LineWidth',1.5)
                        % vert caps
                        plot(c*3-2,r*3-0.5,shp,...
                            'Color','k','MarkerFaceColor',col,'MarkerSize',mksz,'LineWidth',1.5)
                        
                    end
                    
                end
            end
            
            
            subplot(1,3,1)
            ylim([0,6])
            xL = xlim();
            xL(1) = min(-0.001,xL(1));
            xL(2) = max(0.001,xL(2));
            xlim(xL);
            plot([0,0],ylim(),'k:');
            plot(0,startval,'ko','MarkerFaceColor','w','MarkerSize',12)
            
        end
        
        
        % allow user to look through updates with arrow keys
        function updateVisual(EXP,startidx,updatescale)
            if nargin<2 || isempty(startidx)
                startidx = EXP.MES;
            end
            if startidx<0
                startidx = EXP.MES+startidx;
            end
            
            if nargin<3
                updatescale = 0.01;
            end
            
            
            EXP.plotUpdate(startidx,[],updatescale);
            
            while true
                
                k = waitforbuttonpress;
                % 28 leftarrow
                % 29 rightarrow
                % 30 uparrow
                % 31 downarrow
                value = double(get(gcf,'CurrentCharacter'));
                
                switch value
                    case 28
                        startidx = max(1,startidx-1);
                    case 29
                        startidx = min(EXP.MES,startidx+1);
                end
                EXP.plotUpdate(startidx,[],updatescale)
            end
        end
        
        
        % plots the system for a given training step (idx).
        % shows conductances as widths, plots update as color of those edges
        % updatescale < 0 gives max and min conductances for each edge.
        % Only works with testidx = nan (make avg)
        function [updateMap2,voltmap] = plotUpdate(EXP,idx,testidx,updatescale,makefig,targetcolors,doFree,whitesource,graynode)
            buffer = .5;
            if nargin<3
                testidx = [];
            end
            if nargin<8
                if numel(testidx)==1
                    whitesource = true;
                else
                    whitesource = false;
                end
            end
            if nargin<9
                graynode = false;
            end
            if nargin<4 || isempty(updatescale)
                updatescale = 0.01;
            end
            
            
            
            if numel(idx)>1
                error('only for plotting one training step')
            end
            if idx<0
                idx = EXP.MES+idx+1;
            end
            tme = sum(EXP.LearnTimes(1:idx));
            time = cumsum(EXP.LearnTimes);
            
            
            avgtest = false;
            if numel(testidx)==1 && isnan(testidx)
                avgtest = true;
                
                resistances = EXP.state_map(idx,1:EXP.TST);
                maxres = max(resistances,[],4);
                minres = min(resistances,[],4);
                
                if EXP.UPD && EXP.CLA>1
                    % if only selectively updating, jsut show those.
                    [TestClassIDs] = EXP.FoundClassIDs(); % mes x tst
                    wantupdates = TestClassIDs(idx,:)==EXP.TESTCLASSES;
                else
                    wantupdates = true(1,EXP.TST);
                end
                
            end
            
            if nargin<5 || isempty(makefig)
                makefig = true;
            end
            
            
            if nargin<6 || isempty(targetcolors)
                targetcolors = {'k'};
                trgwdth = 1.5;
            else
                trgwdth = 3;
            end
            
            if nargin<7 || isempty(doFree)
                doFree = true;
            end
            
            updatescaleshift =0;
            
            %             if numel(testidx)>4
            %                 error('too many test steps')
            %             end
            
            
            
            
            ctop = [19,47,78]/255;
            cmid = [0,162,255]/255;
            cbot = [1,1,1];
            cmat = [ctop;cmid;cbot];
            
            updateMap = [...
                [linspace(1,1,3)'*cmat(1,1), linspace(1,1,3)'*cmat(1,2), linspace(1,1,3)'*cmat(1,3)];...
                [linspace(cmat(1,1),cmat(2,1),92)', linspace(cmat(1,2),cmat(2,2),92)', linspace(cmat(1,3),cmat(2,3),92)'];...
                [linspace(cmat(2,1),cmat(3,1),92)', linspace(cmat(2,2),cmat(3,2),92)', linspace(cmat(2,3),cmat(3,3),92)'];...
                [linspace(1,1,3)'*cmat(3,1), linspace(1,1,3)'*cmat(3,2), linspace(1,1,3)'*cmat(3,3)];...
                ];
            voltmap = flipud(updateMap);
            
            
            %             ctop = [0.3,1,0.6];
            %             cmid = [1,1,1];
            %             cbot = [0.7,0.2,1];
            %
            
            
            
            % voltmap = hsv(100);
            % voltmap = copper(100);
            voltcolor = @(V) voltmap(round(1+(size(voltmap,1)-1)*V/0.5),:);
            
            
            ctop = [1,0.6,0.3];
            cmid = [1,1,1];
            cbot = [0.7,0.2,1];
            
            cmat = [ctop;cmid;cbot];
            
            
            updateMap = [...
                [linspace(1,1,3)'*cmat(1,1), linspace(1,1,3)'*cmat(1,2), linspace(1,1,3)'*cmat(1,3)];...
                [linspace(cmat(1,1),cmat(2,1),92)', linspace(cmat(1,2),cmat(2,2),92)', linspace(cmat(1,3),cmat(2,3),92)'];...
                [linspace(cmat(2,1),cmat(3,1),92)', linspace(cmat(2,2),cmat(3,2),92)', linspace(cmat(2,3),cmat(3,3),92)'];...
                [linspace(1,1,3)'*cmat(3,1), linspace(1,1,3)'*cmat(3,2), linspace(1,1,3)'*cmat(3,3)];...
                ];
            updateMap = flipud(updateMap);
            
            % cmat(:) = 0.8;
            cmat(2,:) = 0.7;
            
            doingdelta = false;
            if numel(updatescale)>1
                %                 if ~avgtest
                %                     error("must do avgtest: testidx==nan, for capacitor update")
                %                 end
                out = EXP.capacitors(updatescale(2))-EXP.capacitors(updatescale(1));
                out = out/(time(updatescale(2))-time(updatescale(1)));
                out = out*10^6;
                doingdelta = avgtest;
                
                
                if ~avgtest
                    
                    out2 = EXP.updateRule(idx,testidx);
                    
                    
                    for tt = 1:size(out2,4)
                        out2dum = out2(:,:,:,tt);
                        out2dum(~isnan(out)) = out(~isnan(out));
                        out2(:,:,:,tt) = out2dum;
                    end
                    
                    out = out2;
                    
                    % size(out)
                    % out2 = EXP.updateRule(idx,testidx);
                    
                end
                
                if numel(updatescale)>2
                    if numel(updatescale)>3
                        updatescaleshift = updatescale(4);
                        
                    end
                    updatescale = updatescale(3);
                else
                    updatescale = .2;
                end
            else
                if avgtest
                    bigout = EXP.updateRule(idx,1:EXP.TST);
                    bigout = bigout(:,:,:,wantupdates);
                    out = mean(bigout,4);
                    
                else
                    out = EXP.updateRule(idx,testidx);
                end
            end
            
            if abs(updatescaleshift)>=1000
                cmat(1,:) = [0,118,186]/255*0+.7;
            end
            updateMap2 = [...
                [linspace(1,1,3)'*cmat(1,1), linspace(1,1,3)'*cmat(1,2), linspace(1,1,3)'*cmat(1,3)];...
                [linspace(cmat(1,1),cmat(2,1),92)', linspace(cmat(1,2),cmat(2,2),92)', linspace(cmat(1,3),cmat(2,3),92)'];...
                [linspace(cmat(2,1),cmat(3,1),92)', linspace(cmat(2,2),cmat(3,2),92)', linspace(cmat(2,3),cmat(3,3),92)'];...
                [linspace(1,1,3)'*cmat(3,1), linspace(1,1,3)'*cmat(3,2), linspace(1,1,3)'*cmat(3,3)];...
                ];
            updateMap2 = flipud(updateMap2);
            
            caps = EXP.capacitors(idx);
            
            if ~EXP.ISPERIODIC(1)
                out = out(1:end-2,:,:,:);
            end
            
            if ~EXP.ISPERIODIC(2)
                out = out(:,1:end-2,:,:);
            end
            
            
            if size(out,4)>1
                plts = size(out,4)+1;
            else
                plts = 1;
            end
            
            if makefig
                figure(2211011);
                clf
                
            end
            
            PLTS = plts;
            widen = 0.01;
            for k = 1:plts
                if makefig || PLTS>1
                    subplot(1,PLTS,k );
                end
                if k < plts
                    img = out(:,:,1,k);
                else
                    img = mean(out(:,:,1,:),4);
                end
                img(isnan(img)) = 0;
                
                % imagesc(real(img)*0);
                caxis([-1,1]*abs(updatescale) + updatescaleshift)
                colormap(gca,updateMap2)
                axis equal
                axis off
                
                if k == plts
                    if isempty(testidx)
                        title(['Training step t=',num2str(idx),', ',num2str(round(tme/1000,1)),'ms'])
                    else
                        title(['Mean for t=',num2str(idx),', ',num2str(round(tme/1000,1)),'ms'])
                    end
                    if updatescale < 10
                        colorbar();
                    end
                end
                
                if doFree>0
                    getvolt = @(inval) imag(inval);
                elseif doFree<0
                    getvolt = @(inval) real(inval)-imag(inval);
                else
                    getvolt = @(inval) real(inval);
                end
                
                maxwdth = 18;
                minwdth = 0.1;
                condpow = .75;
                S = size(updateMap2,1);
                updateColor= @(val) updateMap2(round(max(1,min(S, 1+(S-1)*((val+updatescale-updatescaleshift)/(2*updatescale))))),:);
                
                for r0 = 1:4
                    yval = r0*3-2;
                    for c0 = 1:4
                        xval = c0*3-2;
                        hold on
                        
                        aV = getvolt(img(r0*3-2,c0*3-2));
                        
                        if r0<4 || EXP.ISPERIODIC(1)
                            % vertical edge
                            
                            nextrow = r0*3+1;
                            if nextrow>size(img,1)
                                nextrow = 1;
                            end
                            bV = getvolt(img(nextrow,c0*3-2));
                            
                            
                            if avgtest && updatescale<0
                                condmin = 1./maxres(yval+1,xval);
                                condmax = 1./minres(yval+1,xval);
                                cond = [condmax,condmin];
                                
                                clr = [1,1,1]*0.2;
                                %                                 cond =  EXP.calc_conductance2(aV,bV,caps(yval+1,xval));
                                %                                 clr = 1-(max(log10(condmax/condmin),.1))/1.5;
                                %                                 clr = clr* [1,1,1]*.5 + (1-clr)*cbot;
                            else
                                cond =  EXP.calc_conductance3(aV,bV,caps(yval+1,xval));
                                
                                clr = updateColor(img(yval+1,xval));
                                
                            end
                            wdth =  max(min(.9*(cond.^condpow)*10^3.2,maxwdth),minwdth);
                            
                            
                            
                            for kk = 1:length(cond)
                                if r0==4
                                    plot(xval*[1,1],[yval,yval+1.75],'-','Color',clr+(kk-1)*0.5,'LineWidth',wdth(kk))
                                    plot(xval*[1,1],[-.75,1],'-','Color',clr+(kk-1)*0.5,'LineWidth',wdth(kk))
                                else
                                    plot(xval*[1,1],[yval,yval+3],'-','Color',clr+(kk-1)*0.5,'LineWidth',wdth(kk))
                                end
                            end
                            
                            
                        end
                        if c0<4 || EXP.ISPERIODIC(2)
                            % horizontal edge
                            
                            
                            nextcol = c0*3+1;
                            if nextcol>size(img,2)
                                nextcol = 1;
                            end
                            bV = getvolt(img(r0*3-2,nextcol));
                            
                            
                            
                            if avgtest && updatescale<0
                                condmin = 1./maxres(yval,xval+1);
                                condmax = 1./minres(yval,xval+1);
                                cond = [condmax,condmin];
                                
                                clr = [1,1,1]*0.2;
                                
                                %                                 cond =  EXP.calc_conductance2(aV,bV,caps(yval,xval+1));
                                %                                 clr = 1-(max(log10(condmax/condmin),.1))/1.5;
                                %                                 clr = clr* [1,1,1]*.5 + (1-clr)*cbot;
                            else
                                cond =  EXP.calc_conductance3(aV,bV,caps(yval,xval+1));
                                clr = updateColor(img(yval,xval+1));
                                
                            end
                            wdth =  max(min(.9*(cond.^condpow)*10^3.2,maxwdth),minwdth);
                            
                            
                            for kk = 1:length(cond)
                                if c0==4
                                    plot([xval,xval+1.75],yval*[1,1],'-','Color',clr+(kk-1)*0.5,'LineWidth',round(wdth(kk),1))
                                    plot([-.75,1],yval*[1,1],'-','Color',clr+(kk-1)*0.5,'LineWidth',round(wdth(kk),1))
                                else
                                    plot([xval,xval+3],yval*[1,1],'-','Color',clr+(kk-1)*0.5,'LineWidth',round(wdth(kk),1))
                                end
                            end
                            
                            
                        end
                        
                        
                    end
                end
                
                
                
                
                for r = 1:4
                    for c = 1:4
                        val = getvolt(img(r*3-2,c*3-2));
                        
                        if doingdelta
                            color0 = updateColor(0+updatescaleshift);
                            edgeclr = updateColor(0+updatescaleshift)*0.7;
                        else
                            color0 = voltcolor(val);
                        end
                        
                      
                        
                        hold on
                        shp = 0;
                        lne = '-';
                        wdth = 1.5;
                        rad = 0.75;
                        
                        edgeclr = 'k';
                        
                        xpos = c*3-2;
                        ypos = r*3-2;
                        str = '';
                        
                        [stvals,names] = EXP.nameInputs();
                        % if source
                        sourcematch = and(EXP.SLOC(:,1)==r-1,EXP.SLOC(:,2)==c-1);
                        targmatch = and(EXP.TLOC(:,1)==r-1,EXP.TLOC(:,2)==c-1);
                        if sum(sourcematch)
                            shp = 0;
                            
                            rad = .9;
                            
                            sval = find(sourcematch);
                            str = names{sval};
                            if isreal(stvals(sval)) % if constant
                                if whitesource>0
                                    color0 = [194,49,117]/255;
                                    color0 = [.5,.5,.5];
                                else
                                    color0 = voltcolor(stvals(sval));
                                end
                            else % if variable
                                if whitesource>0
                                    color0 = color0*0 + 1;
                                end
                            end
                            
                            if abs(updatescaleshift)>=1000
                                color0 = color0*0+0.7;
                            end
                            
                            
                            % if target
                        elseif sum(targmatch)
                            tidx = find(targmatch);
                            shp = 0;
                            lne = '-';
                            wdth = trgwdth;
                            
                            if whitesource>0
                                color0 = [194,49,117]/255;
                                color0 = [ 0.8500    0.3250    0.0980];
                                color0 = [242,169,84]/255;
                                if abs(updatescaleshift)>=1000
                                    color0 = [0,118,186]*0+.5;
                                end
                                if tidx==2  && EXP.CLA>1
                                    color0 = [0,118,186]/255;
                                    color0 = [62,149,201]/255;
                                end
                                
                                

                            end
                            rad = .9;
                            
                            edgeclr = targetcolors{mod(tidx-1,length(targetcolors))+1};
                            str = names{tidx+EXP.SOR};
                        else
                              if graynode
                            color0 = color0*0+.8;
                              end
                        end
                        
                        
                        rectangle('LineStyle',lne,'LineWidth',wdth,...
                            'Position',[xpos-rad,ypos-rad,2*rad,2*rad],...
                            'Curvature',[shp,shp],'EdgeColor',edgeclr,'Facecolor',color0);
                        
                        if whitesource>=0
                            h = text(xpos,ypos,str,'Color',whitesource*[1,1,1],'interpreter','latex','HorizontalAlignment','center','VerticalAlignment','middle','FontSize',12,'fontweight','bold');
                        end
                    
                       % if whitesource && ~(strcmp(str,'$V_-$') || strcmp(str,'$V_+$') || strcmp(str,'$V_o$')|| (length(str)>=2 &&strcmp(str(1:2),'$O'))) || (length(str)>2 && strcmp(str(1:2),'$V') && abs(updatescaleshift)>=1000)
                        if whitesource && ~(strcmp(str,'$V_-$') || strcmp(str,'$V_+$') || strcmp(str,'$V_o$')) || (length(str)>2 && strcmp(str(1:2),'$V') && abs(updatescaleshift)>=1000)
                            h.Color = [0,0,0];
                        end
                        
                        
                        
                        
                        %  plot(c*3-2,r*3-2,'ko','MarkerSize',15,...
                        %      'LineWidth',1.5,'MarkerFaceColor',cvolt*val/0.5);
                        
                        xlim([.5,12.5]+(buffer+EXP.ISPERIODIC(2))*[-1,1]-1)
                        ylim([.5,12.5]+(buffer+EXP.ISPERIODIC(1))*[-1,1]-1)
                    end
                end
                
                ax = gca;
                ax.Position = ax.Position + [-widen,0,widen*2,0];
                ax.Color = [1,1,1];
                ax.Visible = true;
                set(gca,'XTick',[],'YTick',[])
                ax.XColor = [1,1,1]*0.8;
                ax.YColor = ax.XColor;
                if k == 1
                    ax1 = ax;
                elseif k == plts
                    ax.Position(3) = ax1.Position(3);
                end
                
                set(gca,'box','on','linewidth',2,'XColor','k','ycolor','k')
                axis ij
            end
            
        end
        
        function [upd,deltas,caps,stds] = badUpdates(EXP,trange)
            
            for t = 1:length(trange)
                if trange(t)<0
                    trange(t) = EXP.MES+1+trange(t);
                end
            end
            
            trange = min(trange):max(trange);
            
            upd = EXP.updateRule(trange,1:EXP.TST);
            upd(imag(upd)~=0) = 0;
            
            caps = EXP.capacitors(trange);
            deltas = caps(:,:,end)-caps(:,:,1);
            
            stds = nanstd(nanmean(upd,4),0,3);
            caps = nanmean(caps,3);
            upd = nanmean(nanmean(upd,4),3);
            
            want = ~isnan(upd);
            %
            %             figure(2301112)
            % %             clf
            % %             subplot(1,3,1)
            % %             imagesc(caps)
            % %             title('cap values (V)')
            % %             colorbar
            % %             caxis([0,5])
            % %
            % %             subplot(1,3,2)
            % %             imagesc(upd)
            % %             title('Correct Update Rule')
            % %             colorbar
            % %             caxis([-.1,.1])
            % %
            % %             subplot(1,3,3)
            % %             imagesc(deltas)
            % %             title('Actual Changes (V)')
            % %             colorbar
            % %             caxis([-1,1]*.1)
            % %
            %
            %
            %             subplot(1,2,1)
            %             errorbar(upd(want),deltas(want),stds(want),'o')
            %
            %
            %             xlabel('Correct Avg Update')
            %             ylabel('Actual Changes (V)')
            %             subplot(1,2,2)
            %             errorbar(caps(want),deltas(want),stds(want),'o')
            %             xlabel('Cap Values (V)')
            %             ylabel('Actual Changes (V)')
            %
            %
            
        end
        
        % generates a 4 x 4 x TrainIdxs x TestIdxs matrix of updates (V^2-V^2)
        % nodes are filled using free state voltages, as imaginary numbers
        % and clamped as real numbers
        function out = updateRule(EXP,idx,testidx)
            S = [4,4];
            idx2 = [2,3,4,1];
            
            if nargin<3
                testidx = [];
            end
            
            if nargin<2||isempty(idx)
                idx = 1:EXP.MES;
            end
            for k = 1:length(idx)
                if idx(k)<0
                    idx(k) = EXP.MES+idx(k)+1;
                end
            end
            
            up = @(a,b,c,d) (a-b).^2 - (c-d).^2;
            out = nan(12,12,length(idx),max(1,length(testidx)));
            for idx0 = 1:length(idx)
                t = idx(idx0); % training step
                for idx1 = 1:max(1,length(testidx))
                    
                    % if no test index, we're doing training
                    if isempty(testidx)
                        
                        VFree = EXP.TrainFreeState(:,:,t)*EXP.NODEMULT;
                        VClamped = EXP.TrainClampedState(:,:,t)*EXP.NODEMULT;
                        % otherwise, we're doing testing
                    else
                        t2 = testidx(idx1);
                        VFree = EXP.TestFreeState(:,:,t,t2)*EXP.NODEMULT;
                        VClamped = EXP.TestClampedState(:,:,t,t2)*EXP.NODEMULT;
                    end
                    
                    for r = 1:S(1)
                        for c = 1:S(2)
                            % node
                            out(r*3-2,c*3-2,idx0,idx1) = 1i*VFree(r,c) + VClamped(r,c);
                            %
                            %                             if r == 4 && c == 2
                            %                                 [VFree(r,c),VFree(idx2(r),c),...
                            %                                 VClamped(r,c),VClamped(idx2(r),c)]
                            %                                 idx2(r)
                            %                             end
                            % vertical edge
                            vertI = up(VFree(r,c),VFree(idx2(r),c),...
                                VClamped(r,c),VClamped(idx2(r),c));
                            
                            out(r*3-(0:1),c*3-2,idx0,idx1) = vertI;
                            % horizontal edge
                            horizI = up(VFree(r,c),VFree(r,idx2(c)),...
                                VClamped(r,c),VClamped(r,idx2(c)));
                            
                            out(r*3-2,c*3-(0:1),idx0,idx1) =horizI;
                        end
                    end
                end
            end
            
        end
        
        function checkUpdates(obj,num)
            shps = {'o','^','v','s'};
            chars = [111, 94, 118, 29];
            
            color = @(C) [C/3,1-C/3,1];
            
            bad_error = 0.05;
            
            
            if nargin<2
                num = 10;
            end
            
            issues = zeros(obj.NSIZE(1),obj.NSIZE(2),2);
            
            
            figure(22070721)
            clf
            
            if num <=0
                vec = 1:find(diff(obj.DOTEST)==1,1,'last');
            else
                vec = unique(round(linspace(1,(length(obj.DOTEST)-1),num)));
            end
            for s = vec
                mult = obj.DOTEST(s+1)-obj.DOTEST(s);
                [H,V] = getTrainUpdate(obj,obj.DOTEST(s),mult);
                V0 = 2*diff(obj.VerticalCapacitors(:,:,s:s+1),1,3);
                H0 = 2*diff(obj.HorizontalCapacitors(:,:,s:s+1),1,3);
                
                % c = [(s-1)/(idx-2),1-(s-1)/(idx-2),1];
                
                for r = 1:size(H,1)
                    for c = 1:size(H,2)
                        
                        ax1 = subplot(2,3,[1,4]);
                        
                        
                        shp = shps{r};
                        color1 = color(c-1);
                        
                        if abs(H0(r,c)-H(r,c))>bad_error
                            issues(r,c,1) = issues(r,c,1)+1;
                        else
                            
                        end
                        
                        
                        plot(H0(r,c),H(r,c),shp,'MarkerFaceColor',color1,'Color',color1/2,'MarkerSize',10)
                        hold on
                        axis equal
                        
                        if abs(V0(r,c)-V(r,c))>bad_error
                            issues(r,c,2) = issues(r,c,2)+1;
                        else
                        end
                        
                        ax2 = subplot(2,3,[2,5]);
                        plot(V0(r,c),V(r,c),shp,'MarkerFaceColor',color1,'Color',color1/2,'MarkerSize',10)
                        hold on
                        axis equal
                        
                        
                    end
                end
                
            end
            
            linkaxes([ax1 ax2],'xy')
            xL = xlim();
            titles = {'Horizontal Edges','Vertical Edges'};
            for k= 1:2
                subplot(2,3,k+[0,3])
                % xlabel('Edge')
                xlabel('Actual Update (V)')
                ylabel('Predicted Update (V)')
                title(titles{k})
                plot(xL,xL,'k-')
                plot(xL,xL+bad_error,'k--')
                plot(xL,xL-bad_error,'k--')
            end
            
            titles = {'Horizontal','Vertical'};
            
            x= 1:obj.NSIZE(2);
            y = 1:obj.NSIZE(1);
            
            for k=1:2
                subplot(2,3,3*k)
                cla
                imagesc(issues(:,:,k))
                title(titles{k})
                caxis([0,max(issues(:))]);
                
                set(gca,'XTick',x,'XTickLabel',x-1,'YTick',y,'YTickLabel',y-1)
                colorbar
                hold on
                for R = 1:obj.NSIZE(1)
                    for C = 1:obj.NSIZE(2)
                        
                        plot(C,R,['k',shps{R}],'MarkerFaceColor',color(C-1),'MarkerSize',12);
                        
                        c = color(C-1);
                        cstr = sprintf('[%s,%s,%s]',num2str(round(c(1),1)),num2str(round(c(2),1)),num2str(round(c(3),1)));
                        
                        if issues(R,C,1)>0
                            disp(['Horizontal Edge ',num2str(R-1),',',...
                                num2str(C-1),' ',char(chars(R)),' ',cstr,...
                                ' ',num2str(issues(R,C,1)),' issues.'])
                        end
                        
                        if issues(R,C,2)>0
                            disp(['Vertical Edge ',num2str(R-1),',',...
                                num2str(C-1),' ',char(chars(R)),' ',cstr,...
                                ' ',num2str(issues(R,C,2)),' issues.'])
                        end
                        
                    end
                end
                
            end
            
        end
        
        
        function showClamping(obj,makefig,doabstime)
            
            T = obj.TLOC + 1; % shift for matlab indexing
            if nargin<2
                makefig = true;
            end
            if makefig
                figure(2207081)
                clf
            end
            if nargin<3 || isempty(doabstime)
                doabstime = false;
            end
            leg = {};
            if doabstime
            time = cumsum(obj.AbsoluteTimes)/1000;
            else
            time = cumsum(obj.LearnTimes)/1000;
            end
            
            
            if sum(obj.ETAS) == 0
                etavec = obj.ETAS + obj.ETA/129;
            else
                etavec = obj.ETAS/129;
            end
            
            MSE0 = obj.TrainingMSE;
            MSE = obj.TestMSE;
            MSE2 = obj.TrainMSE;
            
            for t = 1:size(T,1)
                c = rand(1,3);
                c = (c-min(c))/(max(c)-min(c));
                x = squeeze(obj.TrainFreeState(T(t,1),T(t,2),:));
                y = squeeze(obj.TrainClampedState(T(t,1),T(t,2),:));
                if makefig
                    subplot(4,1,1:2)
                end
                semilogx(time,x,'-','LineWidth',2,'Color',c);
                hold on;
                semilogx(time,y,'--','LineWidth',2,'Color',c/2);
                leg{end+1} = sprintf('Target %d Free',t);
                leg{end+1} = sprintf('Target %d Clamped',t);
                
                if makefig
                    if obj.DEL == 0 || t<=obj.TAR/2
                        subplot(4,1,3)
                        
                        % loglog(time,((x-y)./(etavec')).^2,'-','LineWidth',2,'Color',c);
                        loglog(time,MSE(t,:),'--','LineWidth',2,'Color','k');
                        
                        hold on
                        loglog(time,MSE2(t,:),'-','LineWidth',2,'Color',c);
                        loglog(time,MSE0(t,:),':','LineWidth',2,'Color',c);
                        
                    end
                end
            end
            
            if makefig
                loglog(time,sum(MSE,1),'-','LineWidth',2,'Color','k');
                
                subplot(4,1,1:2)
            end
            
            if makefig
                for t = 1:size(T,1)
                    for k = 1:obj.TRA
                        z = y*0+obj.TRAIN(obj.SOR+t,k);
                        semilogx(time,z,':','LineWidth',1,'Color',0.5*[1,1,1]);
                    end
                end
            end
            makePlotPrettyNow(12)
            
            if makefig
                
                subplot(4,1,1:2)
            end
            xlabel('Training Time (ms)')
            ylabel('Free / Clamped Target Values')
            ylim([0,1])
            legend(leg,'Location','southwest')
            title(obj.printTitle())
            
            if makefig
                subplot(4,1,3)
                xlabel('Training Time (ms)')
                ylabel('Mean Sq. Errors')
                makePlotPrettyNow(12)
                subplot(4,1,4)
                loglog(time,etavec,'-','LineWidth',2,'Color','k');
                xlabel('Training Time (ms)')
                ylabel('\eta')
                ylim([0.1,1.5])
                makePlotPrettyNow(12)
            end
        end
        
        
        function [out,idx] = capacitors(EXP,idx,shift)
            S = size(EXP.HorizontalCapacitors);
            if nargin < 2
                idx = 1:EXP.MES;
            end
            % allow negative indicees to count from back
            for k = 1:length(idx)
                if idx(k)<0
                    idx(k) = EXP.MES+idx(k)+1;
                end
            end
            
            out = nan(S(1)*3, S(2)*3, length(idx));
            for k = 1:length(idx)
                
                for r = 1:S(1)
                    for c = 1:S(2)
                        % vertical capacitor
                        out(r*3-(0:1),c*3-2,k) = EXP.VerticalCapacitors(r,c,idx(k))*EXP.GATEMULT;
                        % horizontal capacitor
                        out(r*3-2,c*3-(0:1),k) = EXP.HorizontalCapacitors(r,c,idx(k))*EXP.GATEMULT;
                    end
                end
                
            end
            
            if nargin >=3
                out = circshift(out,shift(1)*3,1);
                out = circshift(out,shift(2)*3,2);
            end
            
        end
        
        
        function [out,idx] = power_train_state(EXP,idx,shift)
            S = size(EXP.HorizontalCapacitors);
            idx2 = [2,3,4,1];
            
            if nargin < 2
                idx = 1:EXP.MES;
            end
            % allow negative indicees to count from back
            for k = 1:length(idx)
                if idx(k)<0
                    idx(k) = EXP.MES+idx(k)+1;
                end
            end
            
            out = nan(S(1)*3, S(2)*3, length(idx));
            for k = 1:length(idx)
                VCap = EXP.VerticalCapacitors(:,:,idx(k))*EXP.GATEMULT;
                HCap = EXP.HorizontalCapacitors(:,:,idx(k))*EXP.GATEMULT;
                VFree=EXP.TestFreeState(:,:,idx(k))*EXP.NODEMULT;
                
                
                for r = 1:S(1)
                    for c = 1:S(2)
                        
                        % vertical edge
                        out(r*3-(0:1),c*3-2,k) =EXP.calc_power3(VFree(r,c),VFree(idx2(r),c),VCap(r,c));
                        % horizontal edge
                        out(r*3-2,c*3-(0:1),k) =  EXP.calc_power3(VFree(r,c),VFree(r,idx2(c)),HCap(r,c));
                    end
                end
                
            end
            
            if nargin >=3
                out = circshift(out,shift(1)*3,1);
                out = circshift(out,shift(2)*3,2);
            end
            
        end
        
        function [out,quiv,quiv2,trainIdx, testIdx] = full_system(EXP,trainIdx,testIdx,scalings,shift)
            % returns a matrix with voltages (V) and conductances
            
            
            showcond = true; % if false, show currents
            showcharge =false; % if true show gate voltage instead of conductance/current
            
            capZero = 0.4; % shift gate voltage values down by this amount if plotting them
            
            if isempty(EXP.C)
                EXP = EXP.load_power_helpers();
            end
            
            S = size(EXP.HorizontalCapacitors);
            if nargin < 2
                trainIdx = 1:EXP.MES;
            end
            if nargin<3
                testIdx = 1:EXP.TST;
            end
            
            voltages = EXP.TestClampedState(:,:,trainIdx,testIdx);
            %voltages = EXP.TestClampedState(:,:,trainIdx,testIdx);
            quiv = struct();
            quiv.x = cell(length(trainIdx),length(testIdx));
            quiv.y = cell(length(trainIdx),length(testIdx));
            quiv.dx = cell(length(trainIdx),length(testIdx));
            quiv.dy = cell(length(trainIdx),length(testIdx));
            
            quiv2 = struct();
            quiv2.x = [];
            quiv2.dx = [];
            quiv2.y = [];
            quiv2.dy = [];
            
            for k = 1:size(EXP.DIODES,1)
                
                dD = EXP.DIODES(k,3:4)-EXP.DIODES(k,1:2);
                
                quiv2.y(end+1) = (1+EXP.DIODES(k,1))*3-2 + dD(1);
                quiv2.x(end+1) = (1+EXP.DIODES(k,2))*3-2 + dD(2);
                quiv2.dy(end+1) = dD(1);
                quiv2.dx(end+1) = dD(2);
                
                %[quiv2.x(end),quiv2.y(end),quiv2.dx(end),quiv2.dy(end)]
            end
            
            if nargin< 4
                scalings = [10^6,1];
            end
            
            idx2 = [2,3,4,1];
            
            out = nan(S(1)*3, S(2)*3, length(trainIdx),length(testIdx));
            for k = 1:length(trainIdx)
                VCap = EXP.VerticalCapacitors(:,:,trainIdx(k))*EXP.GATEMULT;
                HCap = EXP.HorizontalCapacitors(:,:,trainIdx(k))*EXP.GATEMULT;
                
                for t = 1:length(testIdx)
                    VFree = voltages(:,:,k,t)*EXP.NODEMULT;
                    for r = 1:S(1)
                        for c = 1:S(2)
                            
                            
                            % vertical edge
                            if showcharge
                                out(r*3-(0:1),c*3-2,k,t) = -(VCap(r,c)-capZero)*scalings(1);
                            elseif showcond
                                vertI = EXP.calc_conductance3(VFree(r,c),VFree(idx2(r),c),VCap(r,c));
                                out(r*3-(0:1),c*3-2,k,t) = -vertI*scalings(1);
                            else
                                vertI = EXP.calc_current3(VFree(r,c),VFree(idx2(r),c),VCap(r,c));
                                out(r*3-(0:1),c*3-2,k,t) = -scalings(1)*abs(vertI);
                                
                                quiv.y{k,t}(end+1) = (r-1)*3+2.5 - sign(vertI)*0.5;
                                quiv.x{k,t}(end+1) = c*3-2;
                                quiv.dx{k,t}(end+1) = 0;
                                quiv.dy{k,t}(end+1) = sign(vertI);
                            end
                            
                            
                            
                            
                            % node
                            out(r*3-2, c*3-2,k,t) = VFree(r,c)*scalings(2);
                            
                            
                            % horizontal edge
                            if showcharge
                                out(r*3-2,c*3-(0:1),k,t)=-(HCap(r,c)-capZero)*scalings(1);
                            elseif showcond
                                horizI = EXP.calc_conductance3(VFree(r,c),VFree(r,idx2(c)),HCap(r,c));
                                out(r*3-2,c*3-(0:1),k,t) =-horizI*scalings(1);
                                
                            else
                                
                                horizI = EXP.calc_current3(VFree(r,c),VFree(r,idx2(c)),HCap(r,c));
                                out(r*3-2,c*3-(0:1),k,t) =-scalings(1)*abs(horizI);
                                
                                quiv.y{k,t}(end+1) = (r-1)*3+1;
                                quiv.x{k,t}(end+1) = c*3-0.5 - sign(horizI)*0.5;
                                quiv.dx{k,t}(end+1) = sign(horizI);
                                quiv.dy{k,t}(end+1) = 0;
                            end
                            
                        end
                    end
                end
            end
            
            if nargin >=5
                out = circshift(out,shift(1)*3,1);
                out = circshift(out,shift(2)*3,2);
            end
            
        end
        
        function FullShow(EXP,max_volt,max_curr,pauseVal,saveVid)
            
            time = cumsum(EXP.LearnTimes)/1000;
            
            xor_me = EXP.SOR==4 && EXP.TRA == 4;
            xor_me = false;
            figure(2209061)
            clf
            if nargin<4 || isempty(pauseVal); pauseVal = 0.01; end
            
            if nargin<2 || isempty(max_volt);  max_volt = 0.5; end
            
            if nargin<3 || isempty(max_curr); max_curr = 4; end
            
            if nargin < 5
                saveVid = false;
            end
            
            
            
            if EXP.TST ==4
                subs = [2-(EXP.TST==1),ceil(EXP.TST/2)];
                tvec  = 1:4;
            elseif EXP.CLA>1
                
                subs = [1,EXP.CLA];
                tvec = [];
                for t = 1:EXP.CLA
                    tvec(end+1) = find(EXP.TESTCLASSES==(t-1),1);
                end
            else
                tvec = 1:min(4,EXP.TST);
                subs = [1,tvec(end)];
            end
            
            if xor_me
                subs = [2,3];
            end
            tvec = 1
            subs = [1,1];
            
            x = [...
                [linspace(1,1,3)'*0.3, linspace(1,1,3)'*0.6, linspace(1,1,3)'];...
                [linspace(1,.06,92)'*0.3, linspace(1,.06,92)'*0.6, linspace(1,.06,92)'];...
                [linspace(.06,1,95)'*0.7, linspace(.06,1,95)'*0.2, linspace(.06,1,95)'];...
                [0,0,0]];
            x = flipud(x);
            
            
            x = [...
                0*[linspace(1,1,3)'*0.3, linspace(1,1,3)'*0.6, linspace(1,1,3)'];...
                0*[linspace(1,.06,92)'*0.3, linspace(1,.06,92)'*0.6, linspace(1,.06,92)'];...
                [linspace(.06,1,95)'*0.3, linspace(.06,1,95)'*0.7, linspace(.06,1,95)'*0.2];...
                [0,0,0]];
            x = flipud(x);
            
            
            [out,quiv,quiv2] = EXP.full_system(1:EXP.MES,tvec,1./[max_curr,max_volt]);
            
            if xor_me
                out2 = out;
                out2(out<0) = 0;
                out2 = (sum(out2(:,:,:,1:2),4)-sum(out2(:,:,:,3:4),4));
                out2(out2 > 0.99 ) = 0.99;
                out2(out2 < -0.99 ) = -0.99;
                
            end
            
            
            
            out(out>.99) = .98;
            out(out<-.99) = -.98;
            
            
            savename = [EXP.Directory,'Movies\',EXP.Date];
            num = 1;
            while exist([savename,'__',num2str(num),'.mp4'],'file')
                num = num+1;
            end
            
            if saveVid
                v = VideoWriter([savename,'__',num2str(num),'.mp4'],'MPEG-4');
                v.FrameRate = 5;
                open(v)
            end
            
            
            
            for m = 1:EXP.MES
                for t = 1:length(tvec)
                    subplot(subs(1),subs(2),t+(xor_me * (t>2)))
                    makePlotDefensible2
                    hold off
                    imagesc(out(:,:,m,t))
                    caxis([-1,1])
                    colormap(x);
                    cbh = colorbar();
                    cbh.Ticks = linspace(-1, 1, 3) ; %Create 8 ticks from zero to 1
                    cbh.TickLabels = {[num2str(max_curr*10^6),'muA'], '0', [num2str(max_volt),'V']};
                    
                    % ylabel(['Test Set #',num2str(t)])
                    axis equal; axis tight;
                    set(gca,'XTick',[],'YTick',[]);
                    % title(['Step #',num2str(EXP.DOTEST(m))])
                    title([num2str(round(time(m),2)),'ms'])
                    hold on
                    for s = 1:size(EXP.SLOC,1)
                        plot((EXP.SLOC(s,2))*3+1,(EXP.SLOC(s,1))*3+1,'ko','MarkerSize',12,'MarkerFaceColor','k')
                    end
                    for s = 1:size(EXP.TLOC,1)
                        plot((EXP.TLOC(s,2))*3+1,(EXP.TLOC(s,1))*3+1,'ks','MarkerSize',12,'MarkerFaceColor','k')
                    end
                    headWidth = 5;
                    headLength = 5;
                    xL = xlim;
                    yL = ylim;
                    ah = gca;
                    aPos = ah.Position;
                    ahx = [aPos(1), aPos(1)+aPos(3)];
                    ahy = [aPos(2), aPos(2)+aPos(4)];
                    %                     for idx = 1:length(quiv.x{m,t})
                    %
                    %                         ah = annotation('arrow',...
                    %                             'headStyle','cback1',...
                    %                             'HeadLength',headLength,'HeadWidth',headWidth,...
                    %                             'Color','r');
                    %
                    %
                    %                         set(ah,'parent',gca);
                    %                         set(ah,'Position',[quiv.x{m,t}(idx),quiv.y{m,t}(idx),quiv.dx{m,t}(idx),quiv.dy{m,t}(idx)]);
                    %
                    %                     end
                    if false
                        quiver(quiv.x{m,t},quiv.y{m,t},quiv.dx{m,t},quiv.dy{m,t},...
                            0.35,'LineWidth',2,'Color','k','MaxHeadSize',10);
                    end
                    % Diodes
                    for idx = 1:length(quiv2.x)
                        
                        ah = annotation('arrow',...
                            'headStyle','cback1',...
                            'HeadLength',headLength,'HeadWidth',headWidth,...
                            'Color','r');
                        
                        
                        set(ah,'parent',gca);
                        set(ah,'Position',[quiv2.x(idx),quiv2.y(idx),quiv2.dx(idx),quiv2.dy(idx)]);
                        
                    end
                    
                    makePlotPrettyNow(12,false,1.5)
                    makePlotDefensible2
                end
                
                if xor_me
                    
                    subplot(2,3,[3,6])
                    hold off
                    imagesc(out2(:,:,m)*2)
                    hold on
                    cL = [-1,1];
                    caxis(cL)
                    cbh = colorbar();
                    cbh.Ticks = linspace(cL(1), cL(2), 3) ; %Create 3 ticks from -1 to 1
                    cbh.TickLabels = {['-',num2str(round(max_volt/2,2)),'V'], '0', [num2str(round(max_volt/2,2)),'V']};
                    
                    title('Nonlinearity')
                    axis equal; axis tight;
                    xlabel('Test Set (1+2)-(3+4)')
                    set(gca,'XTick',[],'YTick',[]);
                    makePlotPrettyNow(12,false,1.5)
                    for s = 1:size(EXP.SLOC,1)
                        plot((EXP.SLOC(s,2))*3+1,(EXP.SLOC(s,1))*3+1,'wo','MarkerSize',12,'LineWidth',1.5)
                    end
                    for s = 1:size(EXP.TLOC,1)
                        plot((EXP.TLOC(s,2))*3+1,(EXP.TLOC(s,1))*3+1,'ws','MarkerSize',12,'LineWidth',1.5)
                    end
                    
                end
                
                if m == 0
                    pause(pauseVal*4)
                end
                
                if saveVid
                    
                    frame = getframe(gcf);
                    writeVideo(v,frame);
                else
                    
                    pause(pauseVal)
                end
            end
            
            if saveVid
                close(v)
            end
        end
        
        function [img,h] = plotCapacitors(obj,idx,max_val,shift,fignum)
            
            
            if nargin < 2
                idx = length(obj.LearnTimes);
            end
            
            if idx <0
                idx = obj.MES + idx;
            end
            
            if numel(idx) ~=1 || idx > obj.MES || idx==0
                error('Input must be <= EXP.MES')
            end
            
            if nargin<3
                max_val = 8;
            end
            
            if nargin < 4
                shift = [0,0];
            end
            img = obj.capacitors(idx,shift);
            
            
            
            if nargin<5
                fignum = 2208022;
            end
            figure(fignum);
            h = imagesc(img);
            
            caxis([-.1,.9]*max_val)
            x = colormap();
            x(1:round(size(x,1)/12),:) = 0;
            colormap(x);
            hold on
            for s = 1:size(obj.SLOC,1)
                plot((obj.SLOC(s,2)+shift(2))*3+1,(obj.SLOC(s,1)+shift(1))*3+1,'wo','MarkerFaceColor','w','MarkerSize',12)
            end
            for s = 1:size(obj.TLOC,1)
                plot((obj.TLOC(s,2)+shift(2))*3+1,(obj.TLOC(s,1)+shift(1))*3+1,'ws','MarkerFaceColor',0.5*[1,1,1],'MarkerSize',12)
            end
            title(['After ',num2str(sum(obj.LearnTimes(1:idx)/1000)),'ms learning'])
            
            colorbar;
        end
        
        function CapacitorShow(obj,max_val,pauseVal)
            figure(2207051)
            clf
            if nargin<3
                pauseVal = 0.2;
            end
            
            if nargin<2
                max_val = 8;
            end
            
            for m = 1:obj.MES
                
                
                hold off
                obj.plotCapacitors(m,max_val,[0,0],2207051);
                
                
                title(['Step #',num2str(obj.DOTEST(m))])
                
                
                makePlotPrettyNow(12)
                
                if m == 0
                    pause(pauseVal*4)
                end
                pause(pauseVal)
            end
        end
        
        function plotClassBasis(obj)
            
            figure(2208031)
            clf
            time = cumsum(obj.LearnTimes)/1000;
            leg = {};
            for cla = 1:obj.CLA
                c = rand(1,3);
                c = (c-min(c))/(max(c)-min(c));
                if obj.DEL > 0
                    for t = 1:obj.TAR/2
                        x = squeeze(obj.ClassBasis(t*2 + (-1:0),cla,:));
                        semilogx(time,x(2,:)-x(1,:),'-','LineWidth',2,'Color',c);
                        hold on;
                        
                        leg{end+1} = sprintf('Class %d Targ %d',cla,t);
                    end
                else
                    for t = 1:obj.TAR
                        x = squeeze(obj.ClassBasis(t,cla,:));
                        semilogx(time,x,'-','LineWidth',2,'Color',c);
                        hold on;
                        
                        leg{end+1} = sprintf('Class %d Targ %d',cla,t);
                    end
                end
            end
            xlabel('Training Time (ms)')
            ylabel('Class Basis')
            legend(leg);
            
        end
        
        function [outTest,outTrain] = ClassificationError(obj,idx)
            if obj.CLA<2
                error('not a classification task!')
            end
            
            if isempty(obj.LearnTimes)
                outTest = [];
                outTrain = [];
                return
            end
            obj = CalculateClassificationError(obj);
            
            
            if nargin < 2
                idx = 1:length(obj.LearnTimes);
            end
            idx(idx<0) = obj.MES+idx(idx<0)+1;
            
            
            
            outTrain = zeros(1,length(idx));
            outTest =  zeros(1,length(idx));
            
            for k = 1:length(idx)
                mTrain = obj.TrainConfusion(:,:,idx(k));
                mTest = obj.TestConfusion(:,:,idx(k));
                outTrain(k) = 1- (sum(diag(mTrain))/sum(mTrain(:)));
                outTest(k) = 1- (sum(diag(mTest))/sum(mTest(:)));
            end
            
%             for cla = 1:obj.CLA
%                 
%                 outTrain = outTrain + obj.TrainConfusion(cla,cla,idx)./sum(obj.TrainConfusion(cla,:,idx),2);
%                 outTest = outTest + squeeze(obj.TestConfusion(cla,cla,idx)./sum(obj.TestConfusion(cla,:,idx),2));
%                 
%             end
%             
%             outTrain = 1-squeeze(outTrain/obj.CLA);
%             outTest = 1-squeeze(outTest/obj.CLA);
            
        end
        
        function [testmat,trainmat] = ConfusionMatrix(obj,idx)
            
            if obj.CLA<2
                error('not a classification task!')
            end
            obj = CalculateClassificationError(obj);
            
            if nargin < 2
                idx = 1:length(obj.LearnTimes);
            end
            idx(idx<0) = obj.MES+idx(idx<0)+1;
            
            
            traincounts0 = sum(obj.TrainConfusion(:,:,1),2)';
            traincounts = histcounts(obj.TRAINCLASSES);
            if ~isequal(traincounts,traincounts0)
                error('Problem with classes')
            end
            testcounts = sum(obj.TestConfusion(:,:,1),2)';
            
            trainmat = obj.TrainConfusion(:,:,idx);
            testmat = obj.TestConfusion(:,:,idx);
            
            % CLAxCLAx(MES) (real class, output class)
            for cla = 1:obj.CLA
                
                trainmat(cla,:,:) = trainmat(cla,:,:)/traincounts(cla);
                testmat(cla,:,:) = testmat(cla,:,:)/testcounts(cla);
            end
            
        end
        
        function plotClassificationError(EXP)
            EXP = CalculateClassificationError(EXP);
            
            figure(2208039)
            clf
            time = cumsum(EXP.LearnTimes)/1000;
            leg = {};
            big1 = 0;
            big2 = 0;
            for cla = 1:EXP.CLA
                c = rand(1,3);
                c = (c-min(c))/(max(c)-min(c));
                
                x1 = squeeze(EXP.TrainConfusion(cla,cla,:)./sum(EXP.TrainConfusion(cla,:,:),2));
                semilogx(time,x1,'--','LineWidth',1,'Color',c);
                hold on;
                x2 = squeeze(EXP.TestConfusion(cla,cla,:)./sum(EXP.TestConfusion(cla,:,:),2));
                semilogx(time,x2,'-','LineWidth',1,'Color',c);
                leg{end+1} = ['Train, Class ',num2str(cla)];
                leg{end+1} = ['Test, Class ',num2str(cla)];
                
                big1 = big1 + x1;
                big2 = big2 + x2;
            end
            xlabel('Training Time (ms)')
            ylabel('Accuracy')
            
            semilogx(time,big1/EXP.CLA,'--','LineWidth',2,'Color','k');
            semilogx(time,big2/EXP.CLA,'-','LineWidth',2,'Color','k');
            leg{end+1} = ['Train avg'];
            leg{end+1} = ['Test avg'];
            
            legend(leg,'Location','southeast')
            title(EXP.printTitle());
        end
        
        % Calculate overall and class-specific training and test classification error.
        % Note that overall error is just a simple average over class errors, and does not weight by class population.
        function [TestError, TrainError,TestClassError,TrainClassError] = getClassificationError(EXP)
            EXP = CalculateClassificationError(EXP);
            
            TestClassError = zeros(EXP.CLA,EXP.MES);
            TrainClassError = zeros(EXP.CLA,EXP.MES);
            
            for cla = 1:EXP.CLA
                TrainClassError(cla,:) = squeeze(EXP.TrainConfusion(cla,cla,:)./sum(EXP.TrainConfusion(cla,:,:),2));
                TestClassError(cla,:) = squeeze(EXP.TestConfusion(cla,cla,:)./sum(EXP.TestConfusion(cla,:,:),2));
            end
            
            TestError = zeros(1,EXP.MES);
            TrainError = zeros(1,EXP.MES);
            for t = 1:EXP.MES
                conf = EXP.TestConfusion(:,:,t);
                TestError(t) = 1- (sum(diag(conf))/sum(sum(conf)));
                conf = EXP.TrainConfusion(:,:,t);
                TrainError(t) = 1- (sum(diag(conf))/sum(sum(conf)));
            end
            % TestError = sum(TestClassError,1)/EXP.CLA;
            % TrainError = sum(TrainClassError,1)/EXP.CLA;
        end
        
        function obj = CalculateClassificationError(obj)
            % function has repeat functionality. Simply run FoundClassIDs
            % and suppress outputs.
            [TestClassIDs,TrainClassIDs,TestBuffer,TrainBuffer,obj]= obj.FoundClassIDs();
        end
        
        % buffers have units of 0-1 (not volts)n
        function [TestClassIDs,TrainClassIDs,TestBuffer,TrainBuffer,obj,TestHinge,TrainHinge] = FoundClassIDs(obj,includebuff)
            obj.TestConfusion = zeros(obj.CLA,obj.CLA,obj.MES);
            obj.TrainConfusion = zeros(obj.CLA,obj.CLA,obj.MES);
            
            if nargin<2
                if obj.UPD==4
                    includebuff=3;
                else
                includebuff = 1;
                end
            end
            
            TestClassIDs = zeros(obj.MES,obj.TST);
            TrainClassIDs = zeros(obj.MES,obj.TRA);
            TestBuffer = zeros(obj.MES,obj.TST);
            TrainBuffer = zeros(obj.MES,obj.TRA);
            TestHinge = zeros(obj.MES,obj.TST);
            TrainHinge = zeros(obj.MES,obj.TRA);
            
            if obj.DEL > 0
                basis = obj.ClassBasis(2:2:end,:,:)-obj.ClassBasis(1:2:end,:,:);
                meas = obj.TrainMeasurements(2:2:end,:,:)-obj.TrainMeasurements(1:2:end,:,:); % (TAR)x(TRA)x(MES)
                meas2 = obj.TestMeasurements(2:2:end,:,:)-obj.TestMeasurements(1:2:end,:,:); % (TAR)x(TST)x(MES)
                
                if obj.TAR==2 && obj.HOT>0 &&obj.HOT<1000
               basis2 = basis * 1000;
                
                else
                    basis2= basis;
                end
                
            else
                basis = obj.ClassBasis;% TARxCLAxMES
                meas = obj.TrainMeasurements; % (TAR)x(TRA)x(MES)
                meas2 = obj.TestMeasurements; % (TAR)x(TST)x(MES)
                
            end
            
            
            for idx = 1:obj.MES
                
                
                %training
                for t = 1:obj.TRA
                    cla = obj.TRAINCLASSES(t)+1;
                    
                    
                    idCla = 1;
                    dif = norm(meas(:,t,idx)-basis(:,1,idx));
                    buf = Inf;
                    out = meas(:,t,idx);
                    for k = 2:obj.CLA
                        dif2 = norm(meas(:,t,idx)-basis(:,k,idx));
                        if dif2 < dif
                            buf = min(dif-dif2,buf);
                            dif = dif2;
                            idCla = k;
                        else
                            buf = min(dif2-dif,buf);
                        end
                    end
                    obj.TrainConfusion(cla,idCla,idx) = obj.TrainConfusion(cla,idCla,idx)+1;
                    TrainClassIDs(idx,t) = idCla-1;
                    TrainBuffer(idx,t) = buf;
                    if obj.DEL
                        TrainHinge(idx,t) = out;
                    end
                end
                
                
                
                %test
                for t = 1:obj.TST
                    cla = obj.TESTCLASSES(t)+1;
                    
                    idCla = 1;
                    dif = norm(meas2(:,t,idx)-basis(:,1,idx));
                    buf = Inf;
                    out = meas2(:,t,idx);
                    for k = 2:obj.CLA
                        dif2 = norm(meas2(:,t,idx)-basis(:,k,idx));
                        if dif2 < dif
                            buf = min(dif-dif2,buf);
                            
                            dif = dif2;
                            idCla = k;
                            
                        else
                            buf = min(dif2-dif,buf);
                        end
                    end
                    TestClassIDs(idx,t) = idCla-1;
                    TestBuffer(idx,t) = buf;
                    obj.TestConfusion(cla,idCla,idx) = obj.TestConfusion(cla,idCla,idx)+1;

                end
            end
            

            if obj.HOT>0 && obj.CLA ==2
                HOTVAL = rem(obj.HOT,1000)*(1+(obj.TFB/4));
                
                
                TestOut = permute(meas2,[3,2,1]);
                TrainOut = permute(meas,[3,2,1]);
               % TrainHinge = 2*(TrainClassIDs-0.5).*(TrainBuffer/2);
                labelsTest = (HOTVAL/500)*2*(repmat(obj.TESTCLASSES,size(obj.MES,1),1)-0.5);
                labelsTrain = (HOTVAL/500)*2*(repmat(obj.TRAINCLASSES,size(obj.MES,1),1)-0.5);

                % written this way to handle 0's
                %   hingeerrTRAIN  = (outdiffTRAIN~=0).*(bufferTRAIN-labelsTRAIN*truehinge);
                %  hingeerr  = (outdiff~=0).*(buffer-labels*truehinge);
                labelscale = 1;
                 buffshift = obj.BUF/1000/2;
                if isequal(includebuff,1)
                %EXTRA 2 BC here hinge comes just from output! not buffer calc
                elseif isequal(includebuff,2) % hinge from zero
                    buffshift =  0;
                    labelscale = 0;
                elseif isequal(includebuff,3)
                    labelscale = buffshift/mean(abs(labelsTrain(:))); % hinge from buffer boundary
                else
                    buffshift = 0;
                end
                
                if obj.UPD==4 && ~isequal(includebuff,3)
                   warning('UPD=4 SHOULD USE HINGE FROM BUF')
                end
                TrainHinge  = heaviside(-(TrainOut-sign(labelsTrain)*buffshift).*labelsTrain-eps).*(TrainOut-labelsTrain*labelscale);
                if obj.TST>0
                TestHinge  = heaviside(-(TestOut-sign(labelsTest)*buffshift).*labelsTest-eps).*(TestOut-labelsTest*labelscale);
                end
               % [mean(abs(labelsTrain(:)))*labelscale,buffshift]
            end
        end
        
        
        % Generates string with relevant parameter info
        function str= printTitle(obj)
            if sum(obj.ETAS)>0
                str1 = ['\eta = ',num2str(round(obj.ETAS(1)/129,2)),' to ',num2str(round(obj.ETAS(end)/129,2))];
            else
                str1 =['\eta = ',num2str(round(obj.ETA/129,2))];
            end
            str =[str1,' || \alpha = ',num2str(obj.ALF),' \mus || DUP = ',num2str(obj.DUP),...
                ' || V_{min} = ',num2str(round(obj.VMN*obj.VMNMULT + obj.VMNSHIFT,1)),...
                'V || V_{init} = ',num2str(round(obj.VIT*obj.VITMULT+obj.VITSHIFT,1)),'V'];
            
        end
        function obj = createClassificationFromRegression(obj)
            
            warning('not written yet')
            
        end
        
        function obj = load_power_helpers(obj)
            
            load([network_project_superclass2.Directory,'/MOSFET/22-10-19-15-34-00-fitfunc.mat'],'dGfit','dSfit','Cfit')
            obj.dG = dGfit;
            obj.dS = dSfit;
            obj.C = Cfit;
            if nargout == 0
                error('Must specify same object as output or no effect!')
            end
        end
        
        
        
        function out = calc_current(obj,aV,bV,gateV)
            deltaS = abs(aV-bV);
            deltaG = max(gateV-[aV,bV]);
            out = interpolator(deltaS,deltaG,obj.dS,obj.dG,obj.C);
            if bV>aV
                out = -out;
            end
        end
        
        
        function out = Vt(obj);
            out = 0.62;
        end
        
        function out = S(obj)
            out = 8.5*10^-4; % (Vg-mean(V)-Vt)*S = conductance
        end
        
        
        % Calculate current via a model and fit (by eye) to data
        function [out,state] = calc_current3(EXP,aV,bV,gateV,state)
            Gstar = gateV-EXP.Vt;
            isOhm = Gstar > max(aV,bV);
            
            if nargin<5 % if state is not forced
                isSat = and(Gstar<=max(aV,bV),Gstar>(min(aV,bV)+EXP.SubTransV));
                isSub = ~or(isOhm,isSat);
                state = isSub + 2*isSat + 3*isOhm;
            else
                isOhm = state==3;
                isSat = state==2;
                isSub = state==1;
            end
            
            out = zeros(size(isOhm));
            
            if sum(isOhm)
                Ohm = EXP.Ohmic(aV,bV,gateV);
                out(isOhm) = Ohm(isOhm);
            end
            if sum(isSat)
                Sat = EXP.Sat(aV,bV,gateV);
                out(isSat) = Sat(isSat);
            end
            if sum(isSub)
                Sub = EXP.Sub(aV,bV,gateV);
                out(isSub) = Sub(isSub);
            end
            
            if bV>aV
                out = -out;
            end
        end
        
        function out = calc_conductance3(obj,aV,bV,gateV)
            
            if numel(aV)>1
                out = nan(size(aV));
                for k = 1:numel(aV)
                    out(k) = obj.calc_conductance3(aV(k),bV(k),gateV(k));
                end
            else
                
                aV(aV==bV) = aV(aV==bV)+10^-5; % avoid equal aV, bV situation.
                [out,state] = obj.calc_current3(aV,bV,gateV);
                
                out = abs(out./(aV-bV));
                if state ==1
                    out = 0;
                end
                
            end
        end
        
        function out = calc_power3(obj,aV,bV,gateV)
            if aV==bV
                out = 0;
            else
                out = abs((aV-bV) .* obj.calc_current3(aV,bV,gateV));
            end
        end
        
        function out = calc_resistance3(obj,aV,bV,gateV)
            out = 1./obj.calc_conductance3(aV,bV,gateV);
        end
        
        
        % Calculate conductance via a fit instead of the whole shebang
        % works for matrices too
        function out = calc_conductance2(obj,aV,bV,gateV)
            out = aV*0;
            for k = 1:length(aV)
                if gateV(k)-obj.Vt-(aV(k)+bV(k))/2 >0.2
                    out(k) = (gateV(k) - obj.Vt - (aV(k)+bV(k))/2)*obj.S();
                else
                    out(k) = obj.calc_conductance(aV(k),bV(k),gateV(k));
                end
            end
        end
        
        function out = calc_current2(obj,aV,bV,gateV)
            out = abs(aV-bV)*obj.calc_conductance2(aV,bV,gateV);
        end
        
        
        function out = calc_resistance2(obj,aV,bV,gateV)
            out = 1./obj.calc_conductance2(aV,bV,gateV);
        end
        
        function out = calc_power2(obj,aV,bV,gateV)
            out = (abs(aV-bV).^2) .* obj.calc_conductance2(aV,bV,gateV);
        end
        
        function out = calc_conductance(obj,aV,bV,gateV)
            if aV==bV
                aV = bV+eps;
            end
            deltaS = abs(aV-bV);
            deltaG = max(gateV-[aV,bV]);
            out = interpolator(deltaS,deltaG,obj.dS,obj.dG,obj.C);
            out = out/deltaS;
        end
        function out = calc_resistance(obj,aV,bV,gateV)
            out = 1/obj.calc_conductance(aV,bV,gateV);
        end
        
        
        function out = calc_power(obj,aV,bV,gateV)
            deltaS = abs(aV-bV);
            deltaG = max(gateV-[aV,bV]);
            out = abs(deltaS*interpolator(deltaS,deltaG,obj.dS,obj.dG,obj.C));
        end
        
        function [R,V,I,P,state,OhmI] = calc_edge_state(obj,aV,bV,gateV)
            V = aV-bV;
            [I,state] = obj.calc_current3(aV,bV,gateV);
            
            if state~=3
                OhmI = abs(obj.calc_current3(aV,bV,gateV,3));
            else
                OhmI = I;
            end
            
            
            R = V./I;
            % to correct for dividing by 0, resistance still well defined!
            
            
            
            ep = 10^-4;
            d0 = R>10^10;
            R(d0) = ep./obj.calc_current3(aV(d0)+ep,bV(d0),gateV(d0));
            
            %  R(state==1) = Inf; % 'infinite resistance for state = 1.'
            
            P = I.*V;
        end
        
        function [R,dV,I,P,V,states,OhmI,netI] = state_map(EXP,trainIdx,testIdx,doFree)
            
            
            S = size(EXP.HorizontalCapacitors);
            if nargin < 2 || isempty(trainIdx)
                trainIdx = 1:EXP.MES;
            end
            if nargin<3 || isempty(testIdx)
                testIdx = 1:EXP.TST;
            end
            
            % change from logical indexing if needed.
            if isequal(unique(trainIdx),[0,1])
                trainIdx = find(trainIdx);
            end
            for k = 1:length(trainIdx)
                if trainIdx(k)<0
                    trainIdx(k) = EXP.MES+trainIdx(k)+1;
                end
            end
            if isequal(unique(testIdx),[0,1])
                testIdx = find(testIdx);
            end
            
            if nargin<4
                doFree= true;
            end
            
            
            R = nan(S(1)*3,S(2)*3,length(trainIdx),length(testIdx));
            dV = R;
            I = R;
            P = R;
            V = R;
            states = R;
            OhmI = R;
            netI = zeros(size(R));
            chi = R;
            
            idx2 = [2:S(1),1];
            if S(1) ~=S(2)
                error('Not square!')
            end
            
            for k = 1:length(trainIdx)
                if trainIdx(k)<0
                    trainIdx(k) = EXP.MES+trainIdx(k)+1;
                end
            end
            
            for idx0 = 1:length(trainIdx)
                t = trainIdx(idx0); % training step
                
                VCap0 = EXP.VerticalCapacitors(:,:,t)*EXP.GATEMULT;
                HCap0 = EXP.HorizontalCapacitors(:,:,t)*EXP.GATEMULT;
                
                for idx1 = 1:length(testIdx) % test step
                    VCap = VCap0;
                    HCap = HCap0;
                    if testIdx(idx1) == 0
                        if doFree
                            VState = EXP.TrainFreeState(:,:,t)*EXP.NODEMULT;
                        else
                            VState = EXP.TrainClampedState(:,:,t)*EXP.NODEMULT;
                        end
                    else
                        t2 = testIdx(idx1);
                        if doFree
                            VState = EXP.TestFreeState(:,:,t,t2)*EXP.NODEMULT;
                        else
                            VState = EXP.TestClampedState(:,:,t,t2)*EXP.NODEMULT;
                        end
                        if ~isempty(EXP.VerticalCapacitorsTEST) && ~isempty(EXP.HorizontalCapacitorsTEST)
                            VCap = EXP.VerticalCapacitorsTEST(:,:,t,t2)*EXP.GATEMULT;
                            HCap = EXP.HorizontalCapacitorsTEST(:,:,t,t2)*EXP.GATEMULT;
                        end
                    end
                    for r = 1:S(1)
                        for c = 1:S(2)
                            
                            %node
                            V(r*3-2,c*3-2,idx0,idx1) = VState(r,c);
                            
                            % vertical
                            [res, vol, cur, pwr,state,ohmcur] = EXP.calc_edge_state(VState(r,c),VState(idx2(r),c),VCap(r,c));
                            R(r*3-(0:1),c*3-2,idx0,idx1) = res;
                            dV(r*3-(0:1),c*3-2,idx0,idx1) = abs(vol);
                            I(r*3-(0:1),c*3-2,idx0,idx1) = abs(cur);
                            P(r*3-(0:1),c*3-2,idx0,idx1) = pwr;
                            states(r*3-(0:1),c*3-2,idx0,idx1) = state;
                            OhmI(r*3-(0:1),c*3-2,idx0,idx1) = ohmcur;
                            
                            netI(r*3-2,c*3-2,idx0,idx1) = netI(r*3-2,c*3-2,idx0,idx1)+cur;
                            netI(idx2(r)*3-2,c*3-2,idx0,idx1) = netI(idx2(r)*3-2,c*3-2,idx0,idx1)-cur;
                            
                            
                            % horizontal
                            [res, vol, cur, pwr,state,ohmcur] = EXP.calc_edge_state(VState(r,c),VState(r,idx2(c)),HCap(r,c));
                            R(r*3-2,c*3-(0:1),idx0,idx1) = res;
                            dV(r*3-2,c*3-(0:1),idx0,idx1) = abs(vol);
                            I(r*3-2,c*3-(0:1),idx0,idx1) = abs(cur);
                            P(r*3-2,c*3-(0:1),idx0,idx1) = pwr;
                            states(r*3-2,c*3-(0:1),idx0,idx1) = state;
                            OhmI(r*3-2,c*3-(0:1),idx0,idx1) = ohmcur;
                            
                            netI(r*3-2,c*3-2,idx0,idx1) = netI(r*3-2,c*3-2,idx0,idx1)+cur;
                            netI(r*3-2,idx2(c)*3-2,idx0,idx1) = netI(r*3-2,idx2(c)*3-2,idx0,idx1)-cur;
                            
                        end
                    end
                end
            end
            
            R(isinf(R)) = 10^10;
            R(R<10) = 10;
            
           
        end
        
        function plotchi(EXP,domod)
            
            [out,outmod] = totalchi(EXP);
                ylab = '$\langle | \chi | \rangle $';
            if nargin>=2 && domod
                out = outmod;
                ylab = '$\langle | \chi*K | \rangle $';
            end
            
            figure(2407021)
            clf
            semilogx(cumsum(EXP.LearnTimes)/10^6,out,'ko');
            xlabel('Learning Time (s)')
            ylabel(ylab,'interpreter','latex')
            
        end
        
        function [out,outmod] = totalchi(EXP)
            
           [out,outmod] = EXP.chi();
           out = nanmean(reshape(nanmean(abs(out),4),144,[]),1);
           outmod = nanmean(reshape(nanmean(abs(outmod),4),144,[]),1);
        end
        
        function [out,outmod,orders] = modeChi(EXP,Porder,sourcenum,trainIdx,testIdx)
            
            if nargin < 4 || isempty(trainIdx)
                trainIdx = 1:EXP.MES;
            end
            if nargin<5 || isempty(testIdx)
                testIdx = 1:EXP.TST;
            end
            
            if nargin<2 || isempty(Porder)
                Porder = 2;
            end
            
            if nargin<3
                sourcenum = 1:EXP.SOR;
            end
            
            [out0,outmod0] = chi(EXP,trainIdx,testIdx); %[12,12,MES,TST] (with mods)
            [modes,orders] = makeOrthonormalModes(EXP.TEST(sourcenum,testIdx),Porder);  %[modes, TST] (with mods)
            
            out = zeros(12,12,size(out0,3),size(modes,1));
            outmod = zeros(12,12,size(out0,3),size(modes,1));
            for t = 1:size(out0,3) % meas step
                for m = 1:size(modes,1) % mode
                    for d = 1:size(out,4) % == size(modes,2), datapoints
                        out(:,:,t,m)= out(:,:,t,m) + out0(:,:,t,d)*modes(m,d);
                        outmod(:,:,t,m)= outmod(:,:,t,m) + outmod0(:,:,t,d)*modes(m,d);
                    end
                end
            end
            
            
        end
        
        function [out,outmod] = chi(EXP,trainIdx,testIdx)
            disp('strange thing with first datapoint?')
            
            if EXP.TAR>2 || (EXP.TAR==2 && EXP.DEL <1)
                error('Only set up to calculate chi for single outputs')
            end
            
            if nargin < 2 || isempty(trainIdx)
                trainIdx = 1:EXP.MES;
            end
            
            
            if nargin<3 || isempty(testIdx)
                testIdx = 1:EXP.TST;
            end
            
            % change from logical indexing if needed.
            if isequal(unique(trainIdx),[0,1]) && length(trainIdx)>2
                trainIdx = find(trainIdx);
            end
            if isequal(unique(testIdx),[0,1])&& length(testIdx)>2
                testIdx = find(testIdx);
            end
            
            minsource = min(min(EXP.TRAIN(1:EXP.SOR,:)));
            
            [Rf,dVf,~,~,~,~,~,~    ]= EXP.state_map(trainIdx,1:EXP.TST,true);
            [~ ,dVc,~,~,~,~,~,netIc]= EXP.state_map(trainIdx,1:EXP.TST,false);
            
            out = dVf*nan;
            outI = dVf*0;
            outmod = dVf*nan;
            
            for t2idx = 1:EXP.TST
                netItarg = netIc(EXP.TLOC(1,1)*3+1, EXP.TLOC(1,2)*3+1,:,t2idx);
                if EXP.DEL==1
                    netItarg = netIc(EXP.TLOC(2,1)*3+1, EXP.TLOC(2,2)*3+1,:,t2idx)-netItarg;
                end
                
                for tidx0 = 1:length(testIdx)
                    tidx = testIdx(tidx0);
                    
                    if t2idx~=tidx
                        
                        
                        
                        for q = [.125,.25,.5,1] % only possible q values right now

                            % find q + free states
                            if sum(abs((EXP.TEST(1:EXP.SOR,tidx)-minsource)*(1-q)-(EXP.TEST(1:EXP.SOR,t2idx)-minsource)))<0.00001
                                % t2 is a q state of t (nudged towards total q state)
                                
                               
                                
                                for kidx = 1:length(trainIdx)
                                    
                                    if nansum(nansum(nansum(abs(netItarg(:,:,kidx,:)))))>nansum(nansum(abs(outI(:,:,kidx,tidx0)))) % pick highest injected current for best signal
                                        outI(:,:,kidx,tidx0) =netItarg(:,:,kidx,:);
                                        out(:,:,kidx,tidx0) = -2*dVf(:,:,kidx,tidx).*(dVc(:,:,kidx,t2idx)-dVf(:,:,kidx,tidx).*(1-q))/netItarg(:,:,kidx,:);
                                        outmod(:,:,kidx,tidx0) = out(:,:,kidx,tidx)./Rf(:,:,kidx,tidx);
                                    end
                                    
                                end
                                
                            end
                        end
                    end
                end
                
                
            end
        end

        
        
        
        function [EV,ER,EI] = entropy_map(EXP,trainIdx)
            if nargin<2
                trainIdx = [];
            end
            
            [R,~,I,~,V] = EXP.state_map(trainIdx,1:EXP.TST);
            S = size(R);
            
            
            
            EV = zeros([EXP.NSIZE,S(3)]);
            ER = EV;
            EI = EV;
            
            nb = 8;
            Vbins = linspace(0,EXP.NODEMULT,nb);
            Rbins = logspace(1,9,nb);
            Ibins = linspace(0,EXP.NODEMULT/1000,nb);
            
            for r = 1:EXP.NSIZE(1)
                for c = 1:EXP.NSIZE(2)
                    for t = 1:S(3)
                        
                        rr = r*3-2;
                        cc = c*3-2;
                        
                        
                        % nodes
                        EV(r,c,t) = entropy2(V(rr,cc,t,:),Vbins);
                        
                        % Vertical edges
                        ER(rr+(1:2),cc,t) = entropy2(R(rr+1,cc,t,:),Rbins);
                        EI(rr+(1:2),cc,t) = entropy2(I(rr+1,cc,t,:),Ibins);
                        
                        % Horizontal edges
                        ER(rr,cc+(1:2),t) = entropy2(R(rr,cc+1,t,:),Rbins);
                        EI(rr,cc+(1:2),t) = entropy2(I(rr,cc+1,t,:),Ibins);
                        
                    end
                end
            end
            
            
            
            VR = nan(S(1:3));
            
            nanR = isnan(R);
            nanV = isnan(V);
            allnan = and(nanR,nanV);
            
            R(nanR) = 0;
            V(nanV) = 0;
            I(nanR) = 0;
            VR = V+R;
            VI = V+I;
            VR(allnan) = nan;
            VI(allnan) = nan;
            
            
            
            
        end
        
        function power = calc_state_power(EXP,trainIdx,testIdx)
            
            
            S = size(EXP.HorizontalCapacitors);
            if nargin < 2 || isempty(trainIdx)
                trainIdx = 1:EXP.MES;
            end
            
            trainIdx(trainIdx<0) = trainIdx(trainIdx<0)+EXP.MES+1;
            
            if nargin<3
                testIdx = 1:EXP.TST;
            end
            if isequal(testIdx,0) || isempty(testIdx)
                voltages = EXP.TrainFreeState(:,:,trainIdx);
                testIdx = 1;
            else
                voltages = EXP.TestFreeState(:,:,trainIdx,testIdx);
            end
            
            
            power = zeros(length(trainIdx),length(testIdx));
            
            idx2 = [2,3,4,1];
            if S(1) ~= 4 || S(2) ~= 4
                error('expected 4x4 network')
            end
            
            
            for k = 1:length(trainIdx)
                VCap = EXP.VerticalCapacitors(:,:,trainIdx(k))*EXP.GATEMULT;
                HCap = EXP.HorizontalCapacitors(:,:,trainIdx(k))*EXP.GATEMULT;
                
                for t = 1:length(testIdx)
                    VFree = voltages(:,:,k,t)*EXP.NODEMULT;
                    for r = 1:S(1)
                        for c = 1:S(2)
                            % vertical edge
                            power(k,t) = power(k,t) + EXP.calc_power3(VFree(r,c),VFree(idx2(r),c),VCap(r,c));
                            
                            % horizontal edge
                            power(k,t) = power(k,t) + EXP.calc_power3(VFree(r,c),VFree(r,idx2(c)),HCap(r,c));
                            
                        end
                    end
                    
                    % add source resistance powers
                    
                    
                    if EXP.SOURCERES>0
                        for s = 1:size(EXP.SLOC,1)
                            
                            % [EXP.TEST(s,t)*EXP.NODEMULT,VFree(EXP.SLOC(s,1)+1,EXP.SLOC(s,2)+1)]
                            Vdrop = (EXP.TEST(s,t)*EXP.NODEMULT-VFree(EXP.SLOC(s,1)+1,EXP.SLOC(s,2)+1));
                            %VFree(EXP.SLOC(s,1)+1,EXP.SLOC(s,2)+1)
                            power(k,t) =  power(k,t) +( Vdrop^2)/EXP.SOURCERES;
                            % disp(Vdrop)
                        end
                    end
                    
                end
            end
            
        end
        
        
        
        
        function laplacian = calc_state_laplacian(EXP,trainIdx,testIdx)
            
            
            S = size(EXP.HorizontalCapacitors);
            if nargin < 2 || isempty(trainIdx)
                trainIdx = 1:EXP.MES;
            end
            if nargin<3|| isempty(testIdx)
                testIdx = 1:EXP.TST;
            end
            
            N = prod(EXP.NSIZE);
            laplacian = zeros(N,N,length(trainIdx),length(testIdx));
            
            idx2 = [2,3,4,1];
            if S(1) ~= 4 || S(2) ~= 4
                error('expected 4x4 network')
            end
            voltages = EXP.TestFreeState(:,:,trainIdx,testIdx);
            
            
            
            for k = 1:length(trainIdx)
                VCap = EXP.VerticalCapacitors(:,:,trainIdx(k))*EXP.GATEMULT;
                HCap = EXP.HorizontalCapacitors(:,:,trainIdx(k))*EXP.GATEMULT;
                
                for t = 1:length(testIdx)
                    VFree = voltages(:,:,k,t)*EXP.NODEMULT;
                    for r = 1:S(1)
                        for c = 1:S(2)
                            n = (r-1)*EXP.NSIZE(1)+c;
                            if n>N
                                error('whoops')
                            end
                            n2 = n+EXP.NSIZE(1);
                            n3 = n+1;
                            if n2>N
                                n2 = n2-N;
                            end
                            if n3>N
                                n3 = n3-N;
                            end
                            
                            % vertical edge
                            laplacian(n,n2,k,t) = -EXP.calc_conductance3(VFree(r,c),VFree(idx2(r),c),VCap(r,c));
                            laplacian(n2,n,k,t) = laplacian(n,n2,k,t);
                            
                            % horizontal edge
                            laplacian(n,n3,k,t) = -EXP.calc_conductance3(VFree(r,c),VFree(r,idx2(c)),HCap(r,c));
                            laplacian(n3,n,k,t) = laplacian(n,n3,k,t);
                            
                            
                            laplacian(n,n,k,t) = -laplacian(n,n3,k,t)-laplacian(n,n2,k,t);
                            
                        end
                    end
                    
                end
            end
            
        end
        
        function [powerHoriz,powerVert,out] = calc_internal_current(EXP,trainIdx,testIdx)
            
            
            S = size(EXP.HorizontalCapacitors);
            if nargin < 2
                trainIdx = 1:EXP.MES;
            end
            if nargin<3
                testIdx = 1:EXP.TST;
            end
            
            powerHoriz = zeros(4,4,length(trainIdx),length(testIdx));
            powerVert = zeros(4,4,length(trainIdx),length(testIdx));
            
            out = nan(12,12,length(trainIdx),length(testIdx));
            
            idx2 = [2,3,4,1];
            if S(1) ~= 4 || S(2) ~= 4
                error('expected 4x4 network')
            end
            voltages = EXP.TestFreeState(:,:,trainIdx,testIdx);
            voltages2 = EXP.TestClampedState(:,:,trainIdx,testIdx);
            
            
            for k = 1:length(trainIdx)
                
                for t = 1:length(testIdx)
                    VFree = voltages(:,:,k,t)*EXP.NODEMULT;
                    VClamp = voltages2(:,:,k,t)*EXP.NODEMULT;
                    for r = 1:S(1)
                        for c = 1:S(2)
                            % vertical edge
                            VFdiff =11*(VFree(idx2(r),c)-VFree(r,c));
                            VCdiff = 11*(VClamp(idx2(r),c)-VClamp(r,c));
                            VFbot = VFree(r,c);
                            VCbot = VClamp(r,c);
                            
                            powerVert(r,c,k,t) = abs(VCdiff-VCbot)+abs(VCdiff-VFdiff)+abs(VFdiff-VFbot)+abs(VCbot-VFbot);
                            % horizontal edge
                            
                            out(r*3 + (-1:0),c*3-2,k,t) = powerVert(r,c,k,t);
                            
                            
                            
                            
                            VFdiff = 11*(VFree(r,idx2(c))-VFree(r,c));
                            VCdiff = 11*(VClamp(r,idx2(c))-VClamp(r,c));
                            
                            powerHoriz(r,c,k,t) = abs(VCdiff-VCbot)+abs(VCdiff-VFdiff)+abs(VFdiff-VFbot)+abs(VCbot-VFbot);
                            out(r*3 -2,c*3+(-1:0),k,t) = powerHoriz(r,c,k,t);
                            
                            
                        end
                    end
                    
                end
            end
            powerHoriz = powerHoriz/200;
            powerVert = powerVert/200;
            out = out/200;
            
        end
        
        function [div,div2] = calc_state_divergence(EXP,trainIdx,testIdx)
            
            
            S = size(EXP.HorizontalCapacitors);
            if nargin < 2
                trainIdx = 1:EXP.MES;
            end
            if nargin<3
                testIdx = 1:EXP.TST;
            end
            
            div = zeros(EXP.NSIZE(1),EXP.NSIZE(2),length(trainIdx),length(testIdx));
            div2 = zeros(EXP.NSIZE(1),EXP.NSIZE(2),length(trainIdx),length(testIdx));
            
            idx2 = [2,3,4,1];
            if S(1) ~= 4 || S(2) ~= 4 || sum(EXP.NSIZE~=4)
                error('expected 4x4 network')
            end
            
            
            
            for k = 1:length(trainIdx)
                VCap = EXP.VerticalCapacitors(:,:,trainIdx(k))*EXP.GATEMULT;
                HCap = EXP.HorizontalCapacitors(:,:,trainIdx(k))*EXP.GATEMULT;
                
                for t = 1:length(testIdx)
                    VFree = EXP.TestClampedState(:,:,trainIdx(k),testIdx(t))*EXP.NODEMULT;
                    for r = 1:S(1)
                        for c = 1:S(2)
                            % vertical edge
                            vertI = EXP.calc_current3(VFree(r,c),VFree(idx2(r),c),VCap(r,c));
                            
                            % horizontal edge
                            horizI = EXP.calc_current3(VFree(r,c),VFree(r,idx2(c)),HCap(r,c));
                            
                            
                            div(r,c,k,t) = div(r,c,k,t) + vertI + horizI;
                            div(idx2(r),c,k,t) = div(idx2(r),c,k,t) - vertI;
                            div(r,idx2(c),k,t) = div(r,idx2(c),k,t) - horizI;
                            
                            div2(r,c,k,t) = div2(r,c,k,t) + abs(vertI) + abs(horizI);
                            div2(idx2(r),c,k,t) = div2(idx2(r),c,k,t) + abs(vertI);
                            div2(r,idx2(c),k,t) = div2(r,idx2(c),k,t) +abs(horizI);
                            
                        end
                    end
                end
            end
            
        end
        
        function out = amp_ratio(EXP)
            
            famp = 1 + EXP.RFREE_HIGH/EXP.RFREE_LOW;
            camp = 1 + EXP.RCLAMP_HIGH/EXP.RCLAMP_LOW;
            
            out = camp/famp;
            if out == 0 % if old and wasn't set up... it's 1.
                out = 1;
            end
        end
        
        % plot capacitor values across expeirment.
        %erase flag false does not clear old plot.
        % numtimes is number of time points to histogram
        % if numtimes<0 second plot is cap traces minus smoothing of
        % (-numtimes).
        function h = plotCapTraces(EXP,eraseflag,numtimes)
            
            autolag = EXP.TRA;
            S = size(EXP.HorizontalCapacitors);
            h = figure(2209261);
            if nargin<2 || eraseflag
                clf
            end
            if nargin<2
                eraseflag = Inf;
            end
            
            
            if nargin<3
                numtimes = 0;
            end
            
            time = cumsum(EXP.LearnTimes)/1000; % in ms
            
            H = logspace(0.2,1.2,S(1)+1);
            V = linspace(5.5,3.5,S(2)+1);
            if numtimes~=0
            subplot(2,1,1)
            end
            shps = '^sov';
            cols = 'cbmr';
            cols = {[0,1,1],[0,0,1],[1,0,1],[1,0,0]};
            
            if isempty(time)
                time = 1:EXP.MES;
            end
            for r = 1:S(1)
                for c = 1:S(2)
                    
                    
                    
                    ln = ['-',shps(r)];
                    col = cols{c};
                    % vertical capacitor
                    y = squeeze(EXP.VerticalCapacitors(r,c,:)*EXP.GATEMULT);
                    want = ones(size(time));
                    want(y>eraseflag) = nan;
                    h0 = semilogx(time(:).*want(:),y(:).*want(:),ln,'LineWidth',1,'MarkerFaceColor',col,'Color','k');
                    hold on
                    
                    if r > EXP.VEG
                        h0.MarkerFaceColor = 'none';
                        h0.Color = h0.Color*.25 + 0.75;
                        
                        uistack(h0,'bottom');
                    end
                    plot(H(c)*[1,1],V(r:r+1),'-','LineWidth',2,'Color',col);
                    plot(H(c),mean(V(r:r+1)),shps(r),'Color','k','MarkerFaceColor',col,'MarkerSize',7);
                    
                    
                    % horizontal capacitor
                    x = squeeze(EXP.HorizontalCapacitors(r,c,:)*EXP.GATEMULT);
                    want = ones(size(time));
                    want(x>eraseflag) = nan;
                    h0 = semilogx(time(:).*want(:),x(:).*want(:),ln,'LineWidth',1,'MarkerFaceColor',col*0.5+0.5,'Color',[1,1,1]*0.5);
                    
                    if c > EXP.HEG
                        h0.MarkerFaceColor = 'none';
                        h0.Color = h0.Color*.25 + 0.75;
                        uistack(h0,'bottom');
                    end
                    plot(H(c:c+1),V(r)*[1,1],'-','LineWidth',2,'Color',col*0.5+0.5);
                    plot(mean(H(c:c+1)),V(r),shps(r),'Color',[1,1,1]*0.5,'MarkerFaceColor',col*0.5+0.5,'MarkerSize',7);
                    
                    if numtimes < 0
                        subplot(2,1,2)
                        fy = abs(fft(y-smooth(y,autolag)));
                        fx = abs(fft(x-smooth(x,autolag)));
                        
                        %h0 = plot(1:autolag*3,acf(y-smooth(y,autolag),autolag*3)',ln,'LineWidth',1,'MarkerFaceColor',col,'Color','k');
                        h0 = semilogx(.5*length(fy)./(1:length(fy)/2),fy(1:end/2),ln,'LineWidth',1,'MarkerFaceColor',col,'Color','k');
                        if r > EXP.VEG
                            h0.MarkerFaceColor = 'none';
                            h0.Color = h0.Color*.5 + 0.5;
                            uistack(h0,'bottom');
                        end
                        
                        hold on
                        %h0 = plot(1:autolag*3,acf(x-smooth(x,autolag),autolag*3)',ln,'LineWidth',1,'MarkerFaceColor',col*0.5+0.5,'Color',[1,1,1]*0.5);
                        h0 = semilogx(.5*length(fx)./(1:length(fx)/2),fx(1:end/2),ln,'LineWidth',1,'MarkerFaceColor',col*0.5+0.5,'Color',[1,1,1]*0.5);
                        
                        if c > EXP.HEG
                            
                            h0.MarkerFaceColor = 'none';
                            h0.Color = h0.Color*.5 + 0.5;
                            uistack(h0,'bottom');
                        end
                        subplot(2,1,1)
                        
                        
                    end
                    
                end
            end
            
            
            xlabel('Time (ms)')
            ylabel('Capacitor Gate (V)')
            
            if numtimes<0
                
                set(gca,'xscale','linear')
                subplot(2,1,2)
                
                %  xlabel('Time (measurements)')
                xlabel('Wavelength (measurements)')
                ylabel('Capacitor Gate-smoothing (V)')
                % set(gca,'xscale','linear')
            end
            
            if numtimes>0
                subplot(2,1,2)
                vec = logspace(-.3,.8,12);
                timesteps = round(linspace(1,EXP.MES,numtimes));
                step= diff(vec(1:2))*(numtimes-1)/numtimes;
                
                [y,~] = capacitorHistogram(EXP,vec,numtimes);
                
                leg = {};
                for t0 = 1:numtimes
                    col = ((t0-1)/(numtimes-1)) * [0,1,-1] + [0,0,1];
                    bar(vec-step/2+col(2)*step,y(t0,:),'LineWidth',2,'EdgeColor',col,...
                        'FaceColor','none','BarWidth',1/numtimes)
                    hold on
                    leg{end+1} = [num2str(round(time(timesteps(t0)))),'ms'];
                end
                legend(leg)
                set(gca,'xscale','log')
                xlabel('Capacitor Values (V)')
                ylabel('Count')
            end
            %
        end
        
        % Creates histograms of capacitor values, uses vec as bin centers and
        % returns numtimes histograms, as well as times corresponding
        % (times) to them.
        function [y,times] = capacitorHistogram(EXP,vec,numtimes)
            
            if nargin<2
                vec = 0:.5:8;
            end
            if nargin<3
                numtimes = 10;
            end
            step= diff(vec(1:2))*(numtimes-1)/numtimes;
            
            timesteps = round(linspace(1,EXP.MES,numtimes));
            y = zeros(numtimes,length(vec));
            time = cumsum(EXP.LearnTimes);
            times = time(timesteps);
            
            vec = [-Inf, (vec(1:end-1) + vec(2:end))/2, Inf];
            
            for t0 = 1:numtimes
                t = timesteps(t0);
                caps= [EXP.HorizontalCapacitors(:,:,t);EXP.VerticalCapacitors(:,:,t)];
                y(t0,:) = histcounts(caps(:)*EXP.GATEMULT,vec);
            end
        end
        
        
        function img = capsOverTimeImage(EXP)
            
            
            S =100000;
            time = cumsum(EXP.LearnTimes);
            time = (S+1)*time/max(time);
            fillvec = zeros(1,S);
            idx = 2;
            for k = 1:S
                while time(idx-1)<(k)
                    idx = idx+1;
                end
                fillvec(k) = idx-1;
            end
            
            
            img2 = zeros(4*(EXP.VEG+EXP.HEG),S);
            
            
            
            
            img = zeros(4*(EXP.VEG+EXP.HEG),EXP.MES);
            
            for m = 1:EXP.MES
                a = EXP.HorizontalCapacitors(:,1:EXP.HEG,m)*EXP.GATEMULT;
                b = EXP.VerticalCapacitors(1:EXP.VEG,:,m)*EXP.GATEMULT;
                
                x = sort([a(:);b(:)]);
                img(:,m) = x;
                
                for m2 = find(fillvec==m)
                    img2(:,m2) = x;
                end
                
                
            end
            %img = img;
            
            if nargout==0
                figure(2212122)
                clf
                %                 img2 = repmat(img,1,1,3);
                %
                %
                %                 x = posnegColormap(ceil(size(img,2)/2));
                %                 x = x(1:size(img2,2),:);
                %                 x = x*0+1;
                %                 for rgb = 1:3
                %
                %                 for t = 1:size(img2,2)
                %                     img2(:,t,rgb) = x(t,rgb)*img2(:,t,rgb);
                %                 end
                %                 end
                subplot(2,1,1)
                imagesc(img)
                colormap('bone')
                colorbar
                
                
                colorbar
                subplot(2,1,2)
                imagesc(img2)
                colormap('bone')
                colorbar
                
                %caxis([0,4])
            end
            
        end
        
        function img = nodesOverTimeImage(EXP)
            
            
            S =100000;
            time = cumsum(EXP.LearnTimes);
            time = (S+1)*time/max(time);
            fillvec = zeros(1,S);
            idx = 2;
            for k = 1:S
                while time(idx-1)<(k)
                    idx = idx+1;
                end
                fillvec(k) = idx-1;
            end
            
            img = zeros(16*EXP.TST,EXP.MES);
            
            img2 = zeros(16*EXP.TST,S);
            
            
            for m = 1:EXP.MES
                a = EXP.TestClampedState(:,:,m,:);
                x = sort(a(:));
                img(:,m) = x;
                
                for m2 = find(fillvec==m)
                    img2(:,m2) = x;
                end
                
            end
            
            if nargout==0
                figure(2212123)
                clf
                %                 img2 = repmat(img,1,1,3);
                %
                %
                %                 x = posnegColormap(ceil(size(img,2)/2));
                %                 x = x(1:size(img2,2),:);
                %                 x = x*0+1;
                %                 for rgb = 1:3
                %
                %                 for t = 1:size(img2,2)
                %                     img2(:,t,rgb) = x(t,rgb)*img2(:,t,rgb);
                %                 end
                %                 end
                subplot(2,1,1)
                imagesc(img)
                colormap('copper')
                caxis([0,1])
                colorbar
                subplot(2,1,2)
                imagesc(img2)
                colormap('copper')
                colorbar
                
                caxis([0,1])
                %caxis([0,4])
            end
            
        end
        
        function [val,hv_max,r_max,c_max,t_max] = max_internal_current(EXP)
            
            
            [horiz,vert] = EXP.calc_internal_current;
            
            val = 0;
            for hv = 1:2
                switch hv
                    case 1
                        mat = horiz;
                    case 2
                        mat = vert;
                end
                for r = 1:4
                    for c = 1:4
                        for t = 1:size(mat,3)
                            if mat(r,c,t)>val
                                hv_max = hv;
                                r_max = r;
                                c_max = c;
                                t_max = t;
                                val = mat(r,c,t);
                            end
                        end
                    end
                end
                
            end
            
        end
        
        function stepvec = nodeValueHist(EXP,timesteps,trainflag)
            CLA = EXP.CLA;
            TESTCLASSES = EXP.TESTCLASSES;
            CLASSBASIS = EXP.ClassBasis;
            if EXP.CLA ==1
                CLA = EXP.TST;
                TESTCLASSES = (1:CLA)-1;
                CLASSBASIS = zeros(EXP.TAR,EXP.TST,EXP.MES);
                for m = 1:EXP.MES
                    CLASSBASIS(:,:,m) = EXP.TEST(EXP.SOR+1:end,:,:);
                end
            end
            numcols = 2*CLA;
            
            
            
            lognode = false;
            
            shapes = {'o','s','^','d'};
            colors = {'c','m','b','r','k','y'};
            if nargin<2 || isempty(timesteps)
                timesteps = 4;
            end
            if length(timesteps)==1
                stepvec = round(linspace(1,EXP.MES,timesteps));
            else
                stepvec = timesteps;
                timesteps = length(stepvec);
            end
            
            [R,~,~,~,~] = EXP.state_map(stepvec);
            
            histstep = 0.05;
            
            nodevec = 0:histstep:1;
            
            
            nodeplot = nodevec(2:end)-histstep/2;
            rvec = 0:histstep*10:10;
            rplot = rvec(2:end)-histstep*5;
            
            capvec = 0:(histstep/2):1;
            capplot = capvec(2:end)-histstep/4;
            %             capstep = 0.25;
            %             capvec = 0:step:7;
            %             nodeplot = capvec(2:end)-capstep/2;
            
            
            figure(2301091)
            clf
            timevec = cumsum(EXP.LearnTimes)/1000; % in ms
            timevec = timevec(stepvec);
            maxcount = 0;
            maxcount2 = 0;
            
            maxcounts = zeros(1,timesteps);
            
            for t = 1:timesteps
                
                R2 = R(1:EXP.VEG,1:EXP.HEG,t,:); % last idx is TST
                y2r = histcounts(log10(R2(:)),rvec)/2;
                maxcount2 = max(y2r);
                
                V2 = EXP.TestFreeState(:,:,stepvec(t),:);
                y2v = histcounts(V2(:),nodevec);
                maxcount = max(y2v);
                
                
                for c = 1:CLA
                    want = TESTCLASSES == (c-1);
                    
                    subplot(timesteps,numcols,(t-1)*numcols+c)
                    V = V2(:,:,:,want);
                    y = histcounts(V(:),nodevec);
                    
                    bar(nodeplot*EXP.NODEMULT,y2v,...
                        1,'FaceColor',[1,1,1]*0.9,'EdgeColor','none');
                    hold on
                    bar(nodeplot*EXP.NODEMULT,y,1,...
                        'FaceColor',[1,1,1]*0.8,'EdgeColor',[1,1,1]*0.6);
                    hold on
                    if t==timesteps
                        xlabel('Free Node Voltages (V)')
                    else
                        set(gca,'XTick',[])
                    end
                    if lognode
                        set(gca,'yscale','log')
                    end
                    maxcounts(t) = max(maxcounts(t),max(y));
                    
                    ylabel([num2str(round(timevec(t))),'ms'])
                    
                    
                    
                    
                    
                    subplot(timesteps,numcols,(t-0.5)*numcols+c)
                    
                    %                 C = [EXP.HorizontalCapacitors(:,1:EXP.HEG,stepvec(t))';...
                    %                     EXP.VerticalCapacitors(1:EXP.VEG,:,stepvec(t))]; %#ok<PROPLC>
                    %                 y = histcounts(C(:),capvec);
                    %  bar(capplot*EXP.GATEMULT,y,1);
                    
                    
                    bar(rplot,y2r,1,'FaceColor',[1,1,1]*0.9,...
                        'EdgeColor','none');
                    hold on
                    R3 = R2(:,:,:,want); % last idx is TST
                    
                    y = histcounts(log10(R3(:)),rvec)/2;
                    
                    bar(rplot,y,1,...
                        'FaceColor',[1,1,1]*0.8,'EdgeColor',[1,1,1]*0.6)
                    
                    set(gca,'yscale','log')
                    if t==timesteps
                        xlabel('log_{10}(Resistances) (\Omega)')
                    else
                        set(gca,'XTick',[])
                    end
                    
                end
            end
            
            
            
            voltcolor = voltcolorfcn();
            for t = 1:timesteps
                
                yvec = [.5,maxcounts(t)];
                yvec2 = [.5,maxcount2];
                if ~lognode
                    yvec(1) = 0;
                end
                for c = 1:CLA
                    y = nodevec(2:end)*0;
                    
                    wantidx =  find(TESTCLASSES+1==c);
                    
                    subplot(timesteps,numcols,(t-1)*numcols+c)
                    ylim(yvec)
                    hold on
                    for k = 1:EXP.TAR
                        locs = EXP.TLOC(k,:)+1;
                        
                        % if not classification
                        %                         if EXP.CLA == 1
                        %                             for idx = 1:EXP.TST
                        %
                        %                                 val = EXP.TestFreeState(locs(1),locs(2),stepvec(t),idx)*EXP.NODEMULT;
                        %                                 % [t,k,locs,val]
                        %                                 goalval = EXP.TEST(EXP.SOR+k,idx)*EXP.NODEMULT;
                        %                                 h = plot(goalval*[1,1],yvec,'-','LineWidth',3,'Color',voltcolor(goalval));
                        %                                 uistack(h,'bottom');
                        %                                 uistack(h,'up',1);
                        %
                        %                                 plot(val,yvec(2)*(0.4+0.5*(idx/EXP.TST)),['k',shapes{k}],'MarkerFaceColor',voltcolor(goalval))
                        %                             end
                        %
                        %                             %if classification
                        %                         else
                        %                             if EXP.CLA == 1
                        %
                        %                                goalval = EXP.TEST(EXP.SOR+k,idx)*EXP.NODEMULT;
                        %                             else
                        goalval =CLASSBASIS(k,c,stepvec(t))*EXP.NODEMULT;
                        % end
                        h = plot(goalval,yvec(2),'v','LineWidth',2,...
                            'MarkerSize',10,'MarkerFaceColor',colors{k},'Color','k');
                        %                              h = plot(goalval*[1,1],yvec,'-','LineWidth',4,'Color','k');
                        %                             h = plot(goalval*[1,1],yvec,'-','LineWidth',2,'Color',colors{k});
                        
                        vals = [];
                        for idx0 = 1:length(wantidx)
                            idx = wantidx(idx0);
                            vals(end+1) = EXP.TestFreeState(locs(1),locs(2),stepvec(t),idx);
                            %  plot(val,yvec(2)*(0.4+0.5*(idx0/length(wantidx))),['k',shapes{k}],'MarkerFaceColor',colors{k})
                        end
                        y = y+ histcounts(vals(:),nodevec);
                        
                        
                        h = bar(nodeplot*EXP.NODEMULT,y,1,'FaceColor',colors{k});
                        uistack(h,'down',2*(k)-1);
                        
                        %end
                    end
                    
                    
                end
                subplot(timesteps,numcols,t*numcols)
                ylim(yvec2)
                xlim([0,10])
            end
            
        end
        
        
        function checkMeasurements(EXP,time)
            
            colors = {'b','m','r','k'};
            locs = EXP.TLOC+1;
            
            if nargin<2 || isempty(time)
                time = 1:EXP.MES;
            end
            for k = 1:length(time)
                if time(k)<0
                    time(k) = time(k) + EXP.MES + 1;
                end
            end
            figure(2301111)
            clf
            for idx = 1:EXP.TAR
                x = [];
                y = [];
                
                for t = time
                    
                    x = [x, squeeze(EXP.TestMeasurements(idx,:,t))];
                    y = [y, squeeze(EXP.TestFreeState(locs(idx,1),locs(idx,2),t,:))'];
                    %   x = [x, squeeze(EXP.TrainMeasurements(idx,:,t))];
                    
                    
                    
                end
                shps = {'o','d','^','s','v','>','<'};
                
                for tst = 1:EXP.TST
                    num = 1:length(x);
                    want = rem(num,4) == rem(tst,4);
                    
                    %shift = 0;
                    subplot(1,2,1)
                    plot(x(want),y(want),shps{tst},'Color',colors{idx})
                    
                    % plot(x,circshift(y,shift),'o','Color',colors{idx})
                    xlabel('TestMeasurements')
                    ylabel('TestFreeState')
                    xlim([0,1])
                    ylim([0,1])
                    hold on
                    subplot(1,2,2)
                    plot(num(want),x(want),shps{tst},'Color',colors{idx})
                    hold on
                    plot(num(want),y(want),shps{tst},'Color',colors{idx})
                end
                ylim([0,1])
                xlabel('Measurement #')
                ylabel('Measurement')
            end
            
            
        end
        
        
        function disc = measurementConsistency(EXP)
            
            locs = EXP.TLOC+1;
            
            
            x = [];
            y = [];
            for idx = 1:EXP.TAR
                
                
                for t = 1:EXP.MES
                    
                    x = [x, squeeze(EXP.TestMeasurements(idx,:,t))];
                    y = [y, squeeze(EXP.TestFreeState(locs(idx,1),locs(idx,2),t,:))'];
                    
                    
                end
                
            end
            
            disc = max(abs(x-y));
            
            
            if isempty(disc) % if no test state.
                disc =nan;
            end
        end
        
        % Export all resistors and capacitor used on edges/IO
        function HARDWARE = hardware(EXP)
            
            HARDWARE = struct();
            
            fields = {'RCHARGE','RDRAIN','ROFF',...
                'DIODES','RFREE_LOW','RFREE_HIGH',...
                'RCLAMP_HIGH','RCLAMP_LOW','CCHARGE',...
                'SOURCERES','TARGETRES','CEDGE'};
            
            for k = 1:length(fields)
                HARDWARE.(fields{k}) = EXP.(fields{k});
            end
            
        end
        
        
        
        % returns outputs at each training step (measured) for both free
        % and clamped outputs. Times output are in seconds.
        % out is [targets (tvals),timesteps (idx),2 (f/c)]
        % sourceout is [sources, timesteps]
        function [out,time,sourceout] = trainOutputs(EXP,tvals,idx)
            
            if nargin<2 || isempty(tvals)
                tvals = 1:EXP.TAR;
            end
            if nargin<3 || isempty(idx)
                idx = 1:EXP.MES;
            end
            idx(idx<0) = EXP.MES-idx(idx<0)+1;
            
            out = zeros(length(tvals),length(idx),2);
            
            reshapevec = [1,length(idx),1];
            T = EXP.TLOC + 1;
            for s = 1:length(tvals)
                out(s,:,1) = reshape(EXP.TrainFreeState(T(tvals(s),1),T(tvals(s),2),idx),reshapevec);
                out(s,:,2) = reshape(EXP.TrainClampedState(T(tvals(s),1),T(tvals(s),2),idx),reshapevec);
            end
            
            if nargout>1
                time = cumsum(EXP.LearnTimes)/1000000;
                time = time(idx);
            end
            
            if nargout>2
                sourceout = zeros(EXP.SOR,length(idx));
                
                svals = 1:EXP.SOR;
                
                S = EXP.SLOC + 1;
                for s = 1:length(svals)
                    sourceout(s,:) = reshape(EXP.TrainFreeState(S(svals(s),1),S(svals(s),2),idx),reshapevec(1:2));
                end
            end
        end
        
        
        function plotGapClassification(EXP,doTrain,doTest)
            
            if EXP.DEL ==0
                error('only coded for DEL>0 so far.')
            end
            if EXP.CLA ~=2
                error("just for two classes")
            end
            
            if nargin<3
                doTest = true;
            end
            if nargin<2 || isempty(doTrain)
                doTrain = true;
            end
            
            figure(230504)
            clf
            leg = {};
            
            trainmeas = EXP.NODEMULT*(EXP.TrainMeasurements(2,:,end)-EXP.TrainMeasurements(1,:,end));
            testmeas = EXP.NODEMULT*(EXP.TestMeasurements(2,:,end)-EXP.TestMeasurements(1,:,end));
            if doTrain
                plot(trainmeas(EXP.TRAINCLASSES==0),'ro');
                hold on;
                plot(trainmeas(EXP.TRAINCLASSES==1),'bo');
                leg{end+1} = 'Train Class 1';
                leg{end+1} = 'Train Class 2';
            end
            if doTest
                plot(testmeas(EXP.TESTCLASSES==0),'rs');
                hold on;
                plot(testmeas(EXP.TESTCLASSES==1),'bs');
                leg{end+1} = 'Test Class 1';
                leg{end+1} = 'Test Class 2';
            end
            legend(leg)
            xlabel('T Set #')
            ylabel('Output (\Delta V)')
            makePlotPrettyNow(12)
        end
        
        
        
        function [out,out2] = plotOutputVariance(EXP,dolog,dotest,doplot)
            
            
            
            if nargin<2 || isempty(dolog)
                dolog = true;
            end
            
            if nargin<3 || isempty(dotest)
                dotest = true;
            end
            
            %   if dotest
            out = EXP.TestMeasurements;
            %  else
            out2 = EXP.TrainMeasurements;
            % end
            if EXP.DEL
                out = out(2:2:end,:,:)-out(1:2:end,:,:);
                out2 = out2(2:2:end,:,:)-out2(1:2:end,:,:);
            end
            
            
            out = var(out,0,2); % will be EXP.TAR x EXP.MES
            out = permute(out,[1,3,2]);
            out2 = var(out2,0,2);
            out2 = permute(out2,[1,3,2]);
            
            
            if nargin>=4 && ~doplot
                return
            end
            
            figure(2305092)
            clf
            
            time = cumsum(EXP.LearnTimes)/1000000; % in s
            
            % calculate approx conductance values
            G = EXP.capacitors()-EXP.Vt-0.22; % minus mean voltage
            
            if EXP.HEG == 3
                G(:,11:12,:) = nan;
            end
            if EXP.VEG == 3
                G(11:12,:,:) = nan;
            end
            
            G = reshape(G,144,[]);
            
            for k = 1:EXP.MES
                %  G(:,:,k) = G(:,:,k)/nanmean
                G(:,k) = G(:,k)-G(:,end);
            end
            G = sqrt(squeeze(nansum(G.^2)));
            
            
            subplot(3,1,1)
            leg = {};
            for k = 1:size(out,1)
                loglog(time, out(k,:),'-o','LineWidth',2)
                hold on
                leg{end+1} = ['Target ',num2str(k)];
            end
            
            if ~dotest
                for k = 1:size(out2,1)
                    loglog(time, out2(k,:),'-o','LineWidth',2)
                    hold on
                    leg{end+1} = ['TEST Target ',num2str(k)];
                end
            end
            
            % if EXP.TAR>1
            if size(out,1)>1
                
                loglog(time, sum(out,1),'k-','LineWidth',3)
                leg{end+1} = 'Sum';
                if ~dotest
                    loglog(time, sum(out2,1),'k--','LineWidth',3)
                    leg{end+1} = 'Sum TEST';
                end
            end
            ylabel('Variance of Target')
            legend(leg,'location','southeast')
            
            set(gca,'xticklabel',{},'ytick',logspace(-10,10,21))
            makePlotPrettyNow(12)
            
            subplot(3,1,3)
            loglog(time,G,'-','LineWidth',2)
            ylabel('Cond. Dist to ultimate solution')
            xlabel('Time (s)')
            makePlotPrettyNow(12)
            
            subplot(3,1,2)
            [TestError, TrainError,TestClassError,TrainClassError] = getClassificationError(EXP);
            semilogx(time,TrainError,'r-','LineWidth',2)
            xlabel('Time (ms)')
            if dotest
                ylabel('Train Accuracy')
            else
                hold on
                semilogx(time,TestError,'b-','LineWidth',2)
                
                ylabel('Accuracy')
                legend({'Train Error','Test Error'})
            end
            yL = ylim();
            ylim([min(.5,yL(1)),1])
            makePlotPrettyNow(12)
            
            
            if ~dolog
                for k = 1:3
                    subplot(3,1,k)
                    set(gca,'xscale','lin')
                end
            end
            
        end
        
        
        function out = C0(EXP)
            out =  0.52*10^-3; % in A/V^2 from calculations and spec sheet
            %            out = out*1.55; % from matching with fit
            %            out=out*0.56;
            out = out*1.32;
            out = 8.5000e-04; % from fits in PNAS paper
        end
        
        function out = lambda(EXP)
            out = 0.02; % in 1/V
        end
        
        % current in ohmic regime
        function I = Ohmic(EXP,aV,bV,gateV)
            I = EXP.C0*((gateV-EXP.Vt-(aV+bV)/2)*abs(aV-bV));
        end
        
        function I = Sat(EXP,aV,bV,gateV)
            I = 0.5*EXP.C0*(gateV-EXP.Vt-min(aV,bV)).^2;
        end
        
        
        function I = Sub(EXP,aV,bV,gateV,Gcrit)
            if nargin<5
                Gcrit = min(aV,bV)+EXP.SubTransV+EXP.Vt;
            end
            
            Icrit = EXP.Sat(aV,bV,Gcrit);
            
            I = Icrit.*10.^((gateV-Gcrit)/EXP.SubDecade);
        end
        
        function out = SubDecade(EXP)
            out = 0.1;
        end
        function out = SubTransV(EXP)
            out = 0.096;
            % (out+x)^2 = 10*(out-x)^2 where x = SubDecade/2
            % essentially matching the slope of Sat region to have smooth
            % transition.
        end
        
        function compareParams(EXP,EXP2)
            isoff = false;
            for k = 1:length(EXP.PARAMNAMES)
                codon = EXP.PARAMNAMES{k};
                if ~isequal(EXP.(codon),EXP2.(codon))
                    disp([codon,': ',num2str(EXP.(codon)),' vs ',num2str(EXP2.(codon))])
                    isoff = true;
                end
            end
            if ~isoff
                disp('All params equal.')
            end
        end
        
        % calculate the error for only the current training datapoint over time
        function [out,trainidxvec] = trainingError(EXP,idx,onlyOn)
            
            
            if nargin>1
                for k = 1:length(idx)
                    if idx(k)<0
                        idx(k) = EXP.MES + idx(k);
                    end
                end
            else
                idx = 1:EXP.MES;
            end
            
            if nargin>2 && onlyOn && length(unique(EXP.TRAINIDX))==1 && EXP.TRAINIDX(1)<0
                % only if 
                trainidxs = rem(EXP.DOTEST-1,-EXP.TRAINIDX(1)*EXP.TRA)+1;
                size(trainidxs)
                keepers = find(EXP.TARGFLAG);
                idx = [];
                for k = keepers
                    idx = [idx,find(trainidxs==k)];
                end
                idx = sort(idx);
               %  train_idx = int((i % (TRA * -TRAINIDX[meas_idx])) / -TRAINIDX[meas_idx]);
            end
            
            
            out = zeros(size(idx));
            trainidxvec = out;
            
            
            sourcestate = zeros(EXP.SOR,length(idx));
            targstate = zeros(EXP.TAR,length(idx));
            answers = EXP.TRAIN((EXP.SOR+1):end,:);
            
            for s = 1:EXP.SOR
                sourcestate(s,:) = squeeze(EXP.TrainFreeState(...
                    EXP.SLOC(s,1)+1,EXP.SLOC(s,2)+1,idx));
            end
            
            for s = 1:EXP.TAR
                targstate(s,:) = squeeze(EXP.TrainFreeState(...
                    EXP.TLOC(s,1)+1,EXP.TLOC(s,2)+1,idx));
            end
            
            if EXP.DEL>0
                targstate = targstate(2:2:end,:)-targstate(1:2:end,:);
                answers = answers(2:2:end,:)-answers(1:2:end,:);
            end
            
            for k = 1:length(idx)
                trainidxvec(k) = EXP.TRAINIDX(idx(k));
                
                if trainidxvec(k)<0
                    trainidxvec(k) = floor(rem(EXP.DOTEST(idx(k))-1,-EXP.TRAINIDX(idx(k))*EXP.TRA)/-EXP.TRAINIDX(idx(k)))+1;
                end
                
                
                % must find trainidx if random
                if trainidxvec(k) == 0
                mindist =Inf;
                
                for t = 1:EXP.TRA
                    d = norm(sourcestate(:,k)-EXP.TRAIN(1:EXP.SOR,t));
                    if d<mindist
                        mindist = d;
                        trainidxvec(k) = t;
                    end
                end
                end
                
                out(k) = sum((targstate(:,k)-answers(:,trainidxvec(k))).^2);

                
            end
            
        end
        
        % generate code (out) and labels (names) for each source and
        % target, depending on their behavior during experiment and DEL
        function [out,names] = nameInputs(EXP)
            
            inputletters = {'V_1','V_2','V_3','V_4','V_5','V_6','V_7','V_8'};
            inputidx = 2;
            
            out = nan(1,EXP.SOR+EXP.TAR);
            names = cell(1,EXP.SOR+EXP.TAR);
            
            TSET = [EXP.TRAIN,EXP.TEST]; % combined sets
            TSET = TSET - min(TSET(:));
            hasVariance = std(TSET(1:EXP.SOR,:),0,2)>10^-10;
            
            % categorize all constant sources
            wantsor = find(~hasVariance);
            for k = wantsor'
                out(k) = mean(TSET(k,:))*EXP.NODEMULT;
                if out(k)>0.25
                    names{k} = '$V_+$';
                elseif out(k)<0.17
                    names{k} = '$V_-$';
                else
                    names{k} = '$V_o$';
                end
            end
            
            isVariable = find(hasVariance);
            if ~isempty(isVariable)
                variableCov = cov(TSET(hasVariance,:)');
                
                
                out(isVariable(1)) = 1i;
                names{isVariable(1)} = ['$',inputletters{1},'$'];
                % categorize variable sources
                for k = 2:length(isVariable)
                    for kk = 1:(k-1)
                        % check correlations with previous variable inputs
                        covcheck = variableCov(k,kk)/variableCov(k,k);
                        if abs(covcheck)>.95 && size(TSET,2)>4 % correlated!
                            if covcheck>0 % same as kk!
                                out(isVariable(k)) = out(isVariable(kk));
                                names{isVariable(k)} = names{isVariable(kk)};
                            else % opposite!
                                out(isVariable(k)) = -out(isVariable(kk));
                                names{isVariable(k)} = ['$',names{isVariable(kk)}(2:end-1),' \prime $'];
                            end
                        end
                    end
                    if isnan(out(isVariable(k))) % no pariing with previous source
                        out(isVariable(k)) = inputidx*1i;
                        names{isVariable(k)} = ['$',inputletters{inputidx},'$'];
                        inputidx = inputidx+1;
                    end
                end
                
            end
            
            
            %             if inputidx == 2 && ~isempty(isVariable) % if just one input, no subscript needed.
            %                 names{isVariable(1)} = ['$',inputletters{inputidx-1}(1),'$'];
            %             end
            
            % categorize outputs
            pm = '-+';
            for t = 1:EXP.TAR
                if EXP.TAR/(1+EXP.DEL) > 1
                    multitarg = true;
                else
                    multitarg = false;
                end
                
                % check for target number and sign (if DEL)
                if EXP.DEL
                    num = 2*ceil(t/2)*(rem(t-1,2)-.5);
                    addstr = ['_',pm(1+(num>0))];
                else
                    num= t;
                    addstr = '';
                end
                
                out(EXP.SOR+t) = num;
                if multitarg
                    addstr = ['_',num2str(abs(num)),addstr];
                end
                names{EXP.SOR+t} = ['$O',addstr,'$'];
                
            end
            
            
        end
        
        % returns mode errors (in voltage quantities).
        function [modeerrors,orders] = modeErrors(EXP,wantvec,test,Porder,targnum,sourcenum)
            
            if nargin<2 || isempty(wantvec)
                wantvec = 1:EXP.MES;
            end
            if nargin<3 || isempty(test)
                test = false;
            end
            if nargin<4 || isempty(Porder)
                Porder = 2;
            end
            if nargin<5 || isempty(targnum)
                targnum = 1:EXP.TAR;
                if EXP.DEL
                    targnum = 1:(EXP.TAR/2);
                end
            end
            if nargin<6
                sourcenum = 1:EXP.SOR;
            end
            wantvec(wantvec<0) = wantvec(wantvec<0)+EXP.MES+1;
            
            EXP = EXP.CalcError;
            
            if test
                INPUTS = EXP.TEST(imag(EXP.nameInputs())~=0,:)*EXP.NODEMULT;
                useError = EXP.TestError*EXP.NODEMULT;
            else
                INPUTS = EXP.TRAIN(imag(EXP.nameInputs())~=0,:)*EXP.NODEMULT;
                useError = EXP.TrainError*EXP.NODEMULT;
            end
            
            if EXP.CLA>1
                [~,~,~,~,~,TestHinge,TrainHinge] = EXP.FoundClassIDs(); % include buffer
                if test
                    useError2 = TestHinge*EXP.NODEMULT;
                else
                    useError2 = TrainHinge*EXP.NODEMULT;
                end
            
                useError = permute(useError2,[3,2,1]);
            end
            
            [modes,orders] = makeOrthonormalModes(INPUTS(sourcenum,:),Porder);
            
            modeerrors = zeros(size(orders,1),length(wantvec),length(targnum));
            

            for tg = 1:length(targnum)
               
            for kk = 0:(Porder*length(sourcenum))
                wantO = find(sum(orders,2)==kk);
                for p = wantO'
                    for t = 1:length(wantvec)
                        modeerrors(p,t,tg) = sum(modes(p,:).*useError(targnum(tg),:,wantvec(t)));
                    end
                end
                
            end
            end
        end
        
        function EXP = SwapNetworks(EXP)
            
            TrainFree = EXP.TrainFreeState;
            EXP.TrainFreeState = EXP.TrainClampedState;
            EXP.TrainClampedState = TrainFree;
            
            TestFree = EXP.TestFreeState;
            EXP.TestFreeState = EXP.TestClampedState;
            EXP.TestClampedState = TestFree;
            
            disp('Network Data Swapped!')
            
        end
        
        function  showDynamics(EXP,meas_per_cycle,timeshift,pausetime,saveVid)
            cc = colororder();
            cc = repmat(cc,5,1);
            if nargin<4 || isempty(pausetime)
                pausetime = 0.2;
            end
            if nargin<5 || isempty(saveVid)
                saveVid = false;
            end
            if nargin<3 || isempty(timeshift)
                timeshift = 0;
            end
            if nargin<2 || isempty(meas_per_cycle)
                meas_per_cycle = EXP.TRA;
            end
            numcycles = EXP.MES/meas_per_cycle;
            
            figure(2405021)
            clf

            
            if saveVid
                savename= EXP.FullPathSaveName;
                savename = ['Movies/movie_',savename(strfind(savename,'experiment_'):end-4)];
                savename = [savename,'_V01'];
                set(gcf,'units','pixels','Position',[ 19   503   1000   300])
                delete([savename,'.mp4']);
                v = VideoWriter([savename,'.mp4'],'MPEG-4');
                v.FrameRate = 1/(max(pausetime,.1));
                v.Quality = 100;
                open(v)
            end
            
            onidx0 = round(find(EXP.TARGFLAG)+timeshift);
            onidx = onidx0*meas_per_cycle/EXP.TRA;
            
            time0 = cumsum(EXP.AbsoluteTimes)/1000;
            T = EXP.TLOC + 1; % shift for matlab indexing
            S = EXP.SLOC + 1; % shift for matlab indexing
            
            subplot(1,3,1)
            %loglog(time0,EXP.trainingError,'-','LineWidth',2,'Color',[1,1,1]*0.7)
            for k = 1:numcycles
                want = (1:meas_per_cycle)+(k-1)*meas_per_cycle;
                
                loglog(mean(time0(want(onidx))),...
                    mean(EXP.trainingError(want(onidx))),...
                    'bx','MarkerSize',10,'LineWidth',2);
                hold on
                 for t = 1:length(onidx)
                 loglog((time0(want(onidx(t)))),...
                    (EXP.trainingError(want(onidx(t)))),...
                    'x','MarkerSize',6,'LineWidth',2,'color',cc(t,:));
                 end
                
            end
            xlabel('Training Time (ms)')
            ylabel('Target Point Error')
            h = plot(time0(onidx),EXP.trainingError(onidx),...
                'ro','MarkerSize',15,'LineWidth',2);
            for k = 1:numcycles
                    subplot(1,3,2:3)
                
                cla
                want = (1:meas_per_cycle)+(k-1)*meas_per_cycle;
                
                time = time0(want);
                time = time-time(1);
                leg = {};
                
                subplot(1,3,1)
                delete(h)
                h = plot(time0(want(onidx)),...
                    EXP.trainingError(want(onidx)),...
                    'ro','MarkerSize',15,'LineWidth',2);
         
                for t = 1:size(T,1)
                    c = cc(t,:);
                    
                    x = squeeze(EXP.TrainFreeState(T(t,1),T(t,2),want));
                    y = squeeze(EXP.TrainClampedState(T(t,1),T(t,2),want));
                    if EXP.DEL>0
                        y2 = squeeze(EXP.TrainClampedState(T(t,1),T(t,2),want(onidx)));
                    else
                        y2 = EXP.TRAIN(EXP.SOR+t,onidx0);
                    end
                    
                    subplot(1,3,2:3)
                    
                    plot(time,x,'-','LineWidth',2,'Color',c);
                    hold on;
                    plot(time,y,'--','LineWidth',2,'Color',c/2);
                    plot(time(onidx),y2,'o','LineWidth',2,'Color',c/2,'MarkerSize',15);
                    leg{end+1} = sprintf('Target %d Free',t);
                    leg{end+1} = sprintf('Target %d Clamped',t);
                    leg{end+1} = sprintf('Target Clamp Pt');
                    
                   % xlim([time(1),time(end)])
                    ylim([0,1])
                    
                  
                end
                
                subplot(1,3,1)
                set(gca,'xtick',logspace(-10,10,21))
                
                for t = 1:size(S,1)
                    subplot(1,3,2:3)
                    x = squeeze(EXP.TrainFreeState(S(t,1),S(t,2),want));
                    y = squeeze(EXP.TrainClampedState(S(t,1),S(t,2),want));
                    
                    plot(time,x,'-','LineWidth',2,'Color','k');
                    hold on;
                    plot(time,y,'--','LineWidth',2,'Color','k');
                    leg{end+1} = sprintf('Source %d Free',t);
                    leg{end+1} = sprintf('Source %d Clamped',t);
                    
                    xlabel('Cycle Time (ms)')
                    
                    
                end
                
                legend(leg,'location','eastoutside')
                
                if saveVid
                    frame = getframe(gcf);
                    writeVideo(v,frame);
                end
                pause(pausetime)
                
            end
            if saveVid
                close(v);
            end
            
        end
        
        
        
        function  showDynamics2(EXP,showpoints,pausetime,saveVid)
            cc = colororder();

            if nargin<2 || isempty(showpoints)
                showpoints = 0;
            end
              if nargin<3 || isempty(pausetime)
                pausetime = 0.2;
            end
            if nargin<4 || isempty(saveVid)
                saveVid = false;
            end
            
            meas_per_cycle = EXP.TRA;
           
            numcycles = EXP.MES/meas_per_cycle;
            
            figure(2502252)
            clf

            
            if saveVid
                savename= EXP.FullPathSaveName;
                savename = ['Movies/movie_',savename(strfind(savename,'experiment_'):end-4)];
                savename = [savename,'_V02'];
                set(gcf,'units','pixels','Position',[ 19   503   1000   300])
                delete([savename,'.mp4']);
                v = VideoWriter([savename,'.mp4'],'MPEG-4');
                v.FrameRate = 1/(max(pausetime,.1));
                v.Quality = 100;
                open(v)
            end
            
            onidx0 = find(EXP.TARGFLAG);
            onidx = round(onidx0*meas_per_cycle/EXP.TRA);
            
            time0 = cumsum(EXP.AbsoluteTimes)/1000;
            traintime0 = cumsum(EXP.LearnTimes)/1000;
            T = EXP.TLOC + 1; % shift for matlab indexing
            S = EXP.SLOC + 1; % shift for matlab indexing
            
            subplot(3,3,[1,4,7])

            for k = 1:numcycles
                want = (1:meas_per_cycle)+(k-1)*meas_per_cycle;
                
                loglog(mean(traintime0(want(onidx))),...
                    mean(EXP.trainingError(want(onidx))),...
                    'ks','MarkerSize',15,'LineWidth',2);
                hold on
                if showpoints
                    for t = 1:length(onidx)
                        loglog((traintime0(want(onidx(t)))),...
                            (EXP.trainingError(want(onidx(t)))),...
                            'x','MarkerSize',6,'LineWidth',2,'color',cc(1,:));
                    end
                end
                
            end
            xlabel('Training Time (ms)')
            ylabel('Error')
            
            if showpoints
                h = plot(traintime0(onidx),EXP.trainingError(onidx),...
                    'ro','MarkerSize',15,'LineWidth',2);
                            legend({'Mean','Individual'},'location','southwest');

            end
            
             
               h2 =  loglog(mean(traintime0(want(onidx))),...
                    mean(EXP.trainingError(want(onidx))),...
                    'ks','MarkerSize',15,'LineWidth',2,'MarkerFaceColor','r');
                
            for k = 1:numcycles
                subplot(3,3,2:3)
                cla
                subplot(3,3,5:6)
                cla
                subplot(3,3,8:9)
                cla
                want = (1:meas_per_cycle)+(k-1)*meas_per_cycle;
                
                time = time0(want);
                time = time-time(1);
                leg0 = {};
                leg1 = {};
                leg2 = {};
                
                subplot(3,3,[1,4,7])
                if showpoints
                    
                    delete(h)
                    h = plot(traintime0(want(onidx)),...
                        EXP.trainingError(want(onidx)),...
                        'ro','MarkerSize',15,'LineWidth',2);
                end
                delete(h2)
                  h2 =  loglog(mean(traintime0(want(onidx))),...
                    mean(EXP.trainingError(want(onidx))),...
                    'ks','MarkerSize',15,'LineWidth',2,'MarkerFaceColor','r');
                for t = 1:(size(T,1)/(1+EXP.DEL))
                    c = cc(t,:);
                    
                  
                    x = squeeze(EXP.TrainFreeState(T(2*t,1),T(2*t,2),want)-EXP.TrainFreeState(T(2*t-1,1),T(2*t-1,2),want));
                    y = squeeze(EXP.TrainClampedState(T(2*t,1),T(2*t,2),want)-EXP.TrainClampedState(T(2*t-1,1),T(2*t-1,2),want));
                   
                    y2 = EXP.TRAIN(EXP.SOR+2*t,onidx0)-EXP.TRAIN(EXP.SOR+2*t-1,onidx0);
                
                    
                    
                    subplot(3,3,2:3)
                    
                    bar(y(onidx)-x(onidx),.2,'LineWidth',.1,'FaceColor',c/2);
                                        hold on;

                    bar(y2-x(onidx)','LineWidth',.1,'FaceColor',c)
                    leg0{end+1} = sprintf('O_%d^C-O_%d^F',t,t);
                    leg0{end+1} = sprintf('L_%d-O_%d^F',t,t);
                  legend(leg0,'location','eastoutside')
                                    xlabel('Datapoint')
ylim([-.2,.2])
                     subplot(3,3,5:6)
                    
                    plot(time,x,'-','LineWidth',2,'Color',c);
                    hold on;
                    plot(time,y,'--','LineWidth',2,'Color',c/2);
                    plot(time(onidx),y2,'o','LineWidth',2,'Color',c/2,'MarkerSize',15);
                    leg1{end+1} = sprintf('Target %d Free',t);
                    leg1{end+1} = sprintf('Target %d Clamped',t);
                    leg1{end+1} = sprintf('Target Clamp Pt');
                    
  ylim([-.5,.5])
                end
                
%                 if EXP.DEL
%                     ylim([-1,1]);
%                 else
%                     ylim([0,1]);
%                 end
                legend(leg1,'location','eastoutside')
            subplot(3,3,[1,4,7])
                set(gca,'xtick',logspace(-10,10,21))
                
                for t = 1:size(S,1)
                    subplot(3,3,8:9)
                    if std(EXP.TRAIN(t,:))>10^-10
                        color = 'k';
                    else
                        color = 0.8*[1,1,1];
                    end
                    x = squeeze(EXP.TrainFreeState(S(t,1),S(t,2),want));
                    y = squeeze(EXP.TrainClampedState(S(t,1),S(t,2),want));
                    
                    plot(time,x,'-','LineWidth',2,'Color',color);
                    hold on;
                    plot(time,y,'--','LineWidth',2,'Color',color);
                    leg2{end+1} = sprintf('Source %d Free',t);
                    leg2{end+1} = sprintf('Source %d Clamped',t);
                    
                end
                xlabel('Cycle (Lab) Time (ms)')
                
                
                legend(leg2,'location','eastoutside')
                
                if saveVid
                    frame = getframe(gcf);
                    writeVideo(v,frame);
                end
                pause(pausetime)
                
            end
            if saveVid
                close(v);
            end
            
        end
        
        
        function [bottoms,tops] = capsOnWalls(EXP,idx)
            
            if nargin<2
                idx = 1:EXP.MES;
            end
            
            if  ~isa(EXP.isout, 'function_handle')
                load('C:\Users\Durian Lab\Documents\MATLAB\Continuous_Network\calibrations\calibration-24-08-06N=10000','isout')
                EXP.isout = isout;
                EXP.save();
            end
            
            
            outs = EXP.isout(EXP.capacitors(idx),EXP.VMN);
            bottoms = squeeze(sum(sum(outs==1)))/2;
            tops = squeeze(sum(sum(outs==-1)))/2;
            
        end
        
        function out = ClampedValues(EXP)
            
            out = zeros(EXP.TAR,EXP.TST,EXP.MES);
            
            for t = 1:EXP.TAR
                for ts = 1:EXP.TST
                    for m = 1:EXP.MES
                        out(t,ts,m) = EXP.TestClampedState(EXP.TLOC(t,1)+1,EXP.TLOC(t,2)+1,m,ts);
                    end
                end
            end
        end
        
        
        function out = ClampedValuesTrain(EXP)
            
            out = zeros(EXP.TAR,EXP.MES);
            
            for t = 1:EXP.TAR
               % for ts = 1:EXP.TRA
                    for m = 1:EXP.MES
                        out(t,m) = EXP.TrainClampedState(EXP.TLOC(t,1)+1,EXP.TLOC(t,2)+1,m);
                    end
              %  end
            end
        end
            
        
        function nums2 = Coefficients(EXP)
            
            
            % extract coefficients
            nums = [];
            idx = 1;
            
            if ~isempty(EXP.Notes)
                while length(nums)<EXP.SOR
                    
                    while idx<length(EXP.Notes) && (isempty(str2num(EXP.Notes(idx))) || isequal('i',EXP.Notes(idx)))&&~strcmp('-',EXP.Notes(idx))
                        idx =idx+1;
                        if length(EXP.Notes)<idx
                            break
                        end
                    end
                    
                    
                    idx2 = idx;
                    while ~isequal(EXP.Notes(idx2),' ') && idx2<length(EXP.Notes)
                        idx2 = idx2+1;
                    end
                    % EXP0.Notes(idx:idx2)
                    nums(end+1) = str2num(EXP.Notes(idx:idx2));
                    idx = idx2+1;
                    
                end
            end
          %  M0 = sum(nums~=0);
            
            [stvals,~] = EXP.nameInputs();
            
            nums2 = nums;
            
            for k = 1:length(nums)
                if real(stvals(k))==0 && imag(stvals(k))<0
                    pairidx = find(stvals==-stvals(k));
                    
                    nums2(pairidx) = nums(pairidx)-nums(k);
                    nums2(k) = 0;
                end
            end
        end
        
        
        function plotHingeError(EXP,minval)
            
            if nargin<2
                minval = 10^-7;
            end
            figure(2501281)
            clf
            [~,~,~,~,~,hingeerr,hingeerrTRAIN] = EXP.FoundClassIDs();
            
            time = cumsum(EXP.LearnTimes)/10^6;
            loglog(time,mean(hingeerrTRAIN.^2,2)+minval,'ko-','LineWidth',2)
            hold on
            loglog(time,mean(hingeerr.^2,2)+minval,'ko--','LineWidth',2)
            xlabel('Time (sec)')
            ylabel('Hinge Loss')
            legend({'Train','Test'})
        end
        
        function plotSourcesAndTargets(EXP)
            
           figure(2502201)
           clf
           time = cumsum(EXP.LearnTimes)/10^6;
           
           vals = {EXP.TrainClampedState,EXP.TrainFreeState};
           names = {'F','C'};
           colors = colororder();
           leg = {};
           for k = 1:2
               for s = 1:EXP.SOR
                    S = EXP.SLOC(s,:)+1;
                    semilogx(time,squeeze(vals{k}(S(1),S(2),:)),'-','color',colors(s,:));
                    hold on
                    leg{end+1} = ['S',num2str(s-1),names{k}];
               end
               for t = 1:EXP.TAR
                    T = EXP.TLOC(t,:)+1;
                    semilogx(time,squeeze(vals{k}(T(1),T(2),:)),'-','color',colors(t+EXP.SOR,:),'LineWidth',2);
                    leg{end+1} = ['T',num2str(s-1),names{k}];
               end
               
            
           end
           
           legend(leg)
        end
        
        
        function L = getTrueLabelScale(EXP)
            if EXP.CLA~=2 || EXP.DEL~=1
                error("This function is meant for CLA=2, DEL=1")
            end
            
            if EXP.UPD==4 % this is using BUF to define label
                 L = EXP.NODEMULT*EXP.BUF/1000/2;
                
            elseif EXP.HOT<=1000
                L = EXP.NODEMULT*EXP.HOT*2/1000;
                if EXP.TFB
                    L = L*5/4;
                end
            else
                error('condition not yet coded')
            end
            

            L = [-L,L];
            
        end
        
    end % end of methods section
    
end

