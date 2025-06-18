function nonlinearClassificationMap2(EXP,num,colors,makefig,fullcolor,sidx,contourbonus)

rigidcolor = false;
onlytest = false;
dobuffer = false;

if nargin<7
    contourbonus = false;
end

if nargin<2
    num = 5;
end
if nargin<4 || makefig
    figure(2303291)
    clf
    makefig = true;
else
    makefig = false;
end


if numel(num) ==1
    if ~makefig
        idx = num;
    else
    idx = unique(round(linspace(2,EXP.MES,num-contourbonus)));
    end
else
    idx = num;
end
idx(idx<0)  = EXP.MES + 1 + idx(idx<0);

[TestClassIDs,TrainClassIDs,TestBuffer,TrainBuffer,EXP,TestOut,TrainOut] = EXP.FoundClassIDs();


if nargin<5 || isempty(fullcolor)
    if EXP.HOT>0
        HOT = EXP.HOT;
        if EXP.UPD == 4
            HOT = EXP.BUF/4;
        end
        if EXP.TFB && HOT<=1000
            HOT = HOT*5/4;
        end
        if HOT > 1000
            fullcolor = rem(HOT,1000)/500;
        else
            fullcolor = HOT/500;
        end
       % fullcolor = fullcolor;
    else
    fullcolor = 0.1;
    end
end

BUF = max(EXP.BUF,1);
numblack = round(BUF/fullcolor/10); % uncertainty region given by EXP.BUF, 200 points total
numblack = min(numblack,99);
numwhite = 2;
if nargin<6 || isempty(sidx)
    sidx = find(std([EXP.TRAIN(1:EXP.SOR,:),EXP.TEST(1:EXP.SOR,:)]')>10^-5,2);
    if length(sidx)<2
        sidx = [1,2];
    end
end

%TestBuffer = TestBuffer;
%TrainBuffer = min(TrainBuffer/fullcolor,1);
TestBuffer = TestBuffer.*(TestClassIDs-0.5);
TrainBuffer = TrainBuffer.*(TrainClassIDs-0.5);



if EXP.CLA==3
    TestBuffer = TestOut;
    TrainBuffer = TrainOut;
end

if nargin<3 || isempty(colors)
    colors = {[62,149,201]/255,[242,169,84]/255,[84,242,169]/255};
end

if length(colors) < 2
    error('2 Colors are needed')
end
if EXP.CLA~=2 && EXP.TAR~=2
    error('only built for binary classification...')
end

if EXP.CLA == 2
    
    clrmap = makeColormap(colors{1},[1,1,1],colors{2});
    for k= 1:size(clrmap,1)
        if rigidcolor
        if k<100
            clrmap(k,:) = colors{1};
        else
            clrmap(k,:) = colors{2};
        end
        end
    end
    for k = (101-numblack):(100+numblack)
        if rigidcolor
        if k <=100
            clrmap(k,:) = colors{1}*0.4+0.6;
        else
            clrmap(k,:) = colors{2}*0.4+0.6;
        end
        end
    end
    clrmap(101-numwhite:100+numwhite,:) = 0;
    
elseif EXP.CLA==3
    clrmap = makeColormap(colors{1},[1,1,1],colors{3},[1,1,1],colors{2});
end
CM = size(clrmap,1);
getclr = @(buf) clrmap( round((buf+1)*(CM-1)/2 +1) ,:);


[x,y] = meshgrid(0:.005:1);
W = size(x,1);


%colors2 = {'none','k'};
wrongclr = 'k';
shapes = {'o','s'};
L = length(idx);

sub2 = max(floor(sqrt(L*.8)),1);
% if L>4
%     sub2 = 2;
% else
%     sub2 = 1;
% end
time = cumsum(EXP.LearnTimes)/1000;

xL = [-1.05,.05]+ [0,1]*W;

for k = 1:L
    M = idx(k);
    if isnan(M)
        continue
    end
    if makefig
        
        subplot(sub2,ceil((L+contourbonus)/sub2),k)
    end
    
    if EXP.TST == 0
        xinterp = EXP.TRAIN(sidx(1),:)';
        yinterp = EXP.TRAIN(sidx(2),:)';
        buff = TrainBuffer(M,:)';
    elseif onlytest
  xinterp = [EXP.TEST(sidx(1),:)]';
        yinterp = [EXP.TEST(sidx(2),:)]';
        buff = [TestBuffer(M,:)]';
  
    else
          xinterp = [EXP.TEST(sidx(1),:),EXP.TRAIN(sidx(1),:)]';
        yinterp = [EXP.TEST(sidx(2),:),EXP.TRAIN(sidx(2),:)]';
        buff = [TestBuffer(M,:),TrainBuffer(M,:)]';
    end
    
    F = scatteredInterpolant(xinterp,yinterp,buff*EXP.NODEMULT) ;
    
    
    imagesc((imgaussfilt((F(x,y)),2)))
    
    
    
    colormap(gca,clrmap)
    caxis(fullcolor*[-1,1]*EXP.NODEMULT)
    hold on
    
    
    
    
    
    %
    %     for t = 1:EXP.TST
    %
    %         plot(EXP.TEST(1,t),EXP.TEST(2,t),'o','Color','none',...
    %                 'MarkerFaceColor',getclr(TestBuffer(M,t)),'MarkerSize',13)
    % hold on
    %     end
    
    
    
    frac = [0,0];
    for c = 1:EXP.CLA
        want0 = EXP.TRAINCLASSES == (c-1);
        want20 = EXP.TESTCLASSES == (c-1);
        
        for c2 = 1:EXP.CLA
            want = and(want0,(c2-1) == TrainClassIDs(M,:));
            want2 = and(want20,(c2-1) == TestClassIDs(M,:));
            
            classColor = colors{c2};
            
            if c==c2
                frac = frac + [sum(want),sum(want2)];
                IDcolor = colors{c2};
            else
                classColor = colors{c2}*.7;
                
                IDcolor = colors{c2}*.7;
            end
            
            
            plot(EXP.TRAIN(sidx(1),want)*W,...
                EXP.TRAIN(sidx(2),want)*W,shapes{1},'Color','none',...
                'MarkerFaceColor',colors{c}*.8,'MarkerSize',5)
            hold on
            
%                         plot(EXP.TEST(sidx(1),want2)*W,...
%                             EXP.TEST(sidx(2),want2)*W,shapes{2},'Color','none',...
%                             'MarkerFaceColor',colors{c})
%                         
        end
    end
    
    %[xy,H] = contour(F(x,y),[0,0]);
    
    %  H.LineWidth = 2;
    %  H.Color = 'k';
    
    if dobuffer
        [xy,H] = contour(F(x,y),BUF*[-1,1]/1000);
        
        H.LineWidth = 2;
        H.Color = [1,1,1]*0.5;
    end
    
    frac = round(100*frac./[EXP.TRA,EXP.TST]);
    if EXP.TST == 0
        tststring = '';
    else
        tststring = [' ',num2str(frac(2)),'%'];
    end
    axis equal
    axis tight
    axis xy
    set(gca,'XTick',[],'YTick',[])
    xlabel('Source 1')
    ylabel('Source 2')
    title([num2str(round(time(M),1)),'ms, ',num2str(frac(1)),'%',tststring])
    xlim(xL)
    ylim(xL)
    
    if contourbonus
        subplot(sub2,ceil((L+contourbonus)/sub2),L+1)
        [xy,H] = contour(F(x,y),[0,0]);
        hold on
        H.LineWidth = 2*(k/L)+1;
        H.Color = (k/L)*[0,-1,-.3]+[.2,1,.9];
        
        axis equal
        axis tight
        axis xy
        set(gca,'XTick',[],'YTick',[])
        xlabel('Source 1')
        ylabel('Source 2')
        xlim(xL)
        ylim(xL)
    end
end
