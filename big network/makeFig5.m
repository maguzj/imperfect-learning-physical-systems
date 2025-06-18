
W = .34;

H = .14;
R1 = 80;
R2 = 50;
theta = -1;

name = {'taskcycle_L_18','taskcycle_L_19'};

%if ~strcmp(EG.Name,strcat(name{:}))
    EG = ExperimentGroup(name,-2:-1);
%end


color = colororder();

colors = {[1,1,1]*.7,color(3,:),color(4,:)};
colors2 = {1.3*[141,27,17]/255, 1.3*[10,30,90]/255};
for k = 1:length(colors2)
    colors2{k} = colors2{k}*0.7+0.3;
end



%%
wantETA = [128,32];
wantNOR = [0,1];
wantALF = [25,200];

want = circshift([ones(1,4),zeros(1,4)],0);
want2 = circshift(want,1);

%want = [want;circshift(want,4)];
tsetkeepers = zeros(size(EG.TRAINSET));
tsetkeepers2 = zeros(size(EG.TRAINSET));
for k = 1:length(EG.trainsetcell)
    if isequal(EG.trainsetcell{k}(end,:),want)
        tsetkeepers(EG.TRAINSET==k) = 1;
    elseif isequal(EG.trainsetcell{k}(end,:),want2)
        tsetkeepers2(EG.TRAINSET==k) = 1;
        
    end
end


diams = [];
etas = EG.ETA;
for k = 1:length(EG.HINGEERR)
    kk = EG.TRAINSET(k);
    diams(k) = EG.trainsetcell{kk}(1,1)-0.5;
end

diams = diams*EG.NODEMULT(1)*1000*2; % really diameter

D = unique(diams);
D = D(1:2:end);
D = sort([D,D]);


[x,y,err,cat] = makeErrorBar(diams,EG.HINGEERRTRAIN*10^6,[],EG.ETA+1i*(EG.NOR+EG.ALF));
err = max((EG.NODEMULT(1)/4096)^2,err);
err = min(err,y-10^-15); % make error bar go below plot axis
[x2,y2,err2,cat2] = makeErrorBar(diams,1-EG.CLASSERRTRAIN,[],EG.ETA+1i*(EG.NOR+EG.ALF));

%%

zeroval = .1;
minlim = zeroval*10^-.25;

bigax = [];

figure(2502031)
clf

subs = reshape(length(wantETA)*length(D)+ [1:length(D)*2],[],2)';
subclumps = {subs(:,1:end/2),subs(:,end/2+1:end)};


leg = {};

for idx = 1:length(wantETA)
    k =  find(cat==(wantETA(idx)+1i*(wantNOR(idx)+wantALF(idx))));
    if ~isempty(k)
        
        linespec = {'s','MarkerSize',10,'LineWidth',2,...
            'Color',colors{idx},'MarkerFaceColor',colors{idx}*0.7+0.3};
        bigax(1)=subplot(length(wantETA)+2,length(D),subclumps{1}(:));
        
        
        errorbar(x,y(:,k),err(:,k),linespec{:});
        hold on
        zerovals = y(:,k) ==0;
        x0 = x(zerovals);
       % plot(x0,x0*0+2*minlim,'v',linespec{4:end},'MarkerSize',8);
        plot(x0,x0*0+zeroval,linespec{:});
%         for jj  = 1:length(x0)
%             text(x0(jj),3*minlim,'0','FontSize',12,'Color',colors{idx},'FontWeight','bold',...
%                 'HorizontalAlignment','center','VerticalAlignment','bottom');
%             
%         end
        bigax(2)=  subplot(length(wantETA)+2,length(D),subclumps{2}(:));
        errorbar(x2,100*(1-y2(:,k)),100*err2(:,k),linespec{:});
        hold on
        leg{end+1} = num2str(cat(k));
    end
end


leg = {'Standard','O.C.'};

subplot(length(wantETA)+2,length(D),subclumps{1}(:))
set(gca,'yscale','log','xscale','lin');
ylabel('Hinge Loss (mV^2)')
tix = logspace(log10(zeroval),3,3-log10(zeroval)+1);

tix2 = cell(size(tix));
for tt = 1:length(tix2)
    tix2{tt} = ['10^{',num2str(log10(tix(tt))),'}'];
end
tix2{1} = '0';

                  
set(gca,'ytick',tix,'yticklabel',tix2)
ylim([minlim,2*10^-3*10^6])
subplot(length(wantETA)+2,length(D),subclumps{2}(:))
ylabel('Classification Error (%)')
set(gca,'yscale','lin','xscale','lin');
ylim([-.04,.55]*100)

%legend(leg,'location','southeast')


xticklabel = cell(size(x2));
for k = 1:length(xticklabel)
    xticklabel{k} = '$r_o$';
    if k >1
        xticklabel{k} =  ['$',num2str(round(x2(k)/x2(1))),xticklabel{k}(2:end)];
    end
end


for k = 1:2
    subplot(length(wantETA)+2,length(D),subclumps{k}(:))
    xlabel('Input Variation \epsilon (mV)')
    makePlotPrettyNow(12)
    % set(gca,'xtick',round(x2,2),'xticklabel',xticklabel,'TickLabelInterpreter','latex')
    xlim([x2(1)-50,x2(end)+50])
    xlim([0,250]*2)
    set(gca,'xtick',(0:50:250)*2)
    if k == 2
        val = 0;
    else
        val = zeroval;
    end
    xL = xlim();
    h = plot(xL,[val,val],'-','linewidth',2,'color',[.3,.3,.3]);
    uistack(h,'bottom')
    
end

%%

manyax = [];

for r = 1:length(wantETA)
    for c = 1:length(D)
        if rem(c,2)==0
            keepers = tsetkeepers2;
        else
            keepers = tsetkeepers;
        end
        manyax(end+1) = subplot(length(wantETA)+2,length(D),(r-1)*length(D)+c);
        k = find(and(and(and(EG.ETA==wantETA(r),and(EG.ALF==wantALF(r),EG.NOR==wantNOR(r))),diams==D(c)),keepers),1);
        try
            nonlinearClassificationMap2(EG.loadExperiment(k),-1,colors2,false);
           hold on
           title('')
            x = get(gca,'children');
            for kk = 1:length(x)
                try
                    x(kk).MarkerSize = 6;
                    x(kk).Color = 'k';
                    x(kk).LineWidth = 0.5;
                    
                catch
                end
            end
        catch
        end
          if r == 2 && rem(c,2)==0
            if c == 2
                str0 = '$\epsilon=';
            else
                str0 = '$';
            end
            text(5,5,[str0,num2str(round(D(c))),'$mV'],'FontSize',12,'Color','w','FontWeight','bold',...
                'HorizontalAlignment','left','VerticalAlignment','bottom','interpreter','latex');
          end
          if r == 2 && c == 2
            quiver(100+R1*sin(theta),100-R1*cos(theta),-R2*sin(theta),R2*cos(theta),'w','LineWidth',2,'maxheadsize',10)
            quiver(100-R1*sin(theta),100+R1*cos(theta),R2*sin(theta),-R2*cos(theta),'w','LineWidth',2,'maxheadsize',10)
        end
        
      
        % plot(10,10,[colors{bignonlin(k)+1},shapes{(EG.UPD(k)==4)+1}],'LineWidth',2,'markersize',(log(rads(k))+3.5)*4);
        % hold on;
        makePlotPrettyNow(12);
    end
end



%%
set(gcf,'units','pixels','position',[10,300,600,600])



for k = 1:2
    axes(bigax(k))
    set(gca,'innerposition',[(W+.18)*(2-k)+.09,.1+(2.7-length(wantALF))*(H+.01),W,W])
end


manyax2 = reshape(manyax,2,[])';
manyax2 = manyax2(:);

idx=0;

theta = -1.4*pi/4;
for r = 1:length(wantETA)
    for c = 1:length(D)
        idx = idx+1;
        axes(manyax2(idx));
        
        set(gca,'innerposition',[(H+.01)*(c-1)+.05+.02*(c>3),.97-(H+.01)*r,H,H])
        
        
        
        if r~= length(wantETA)
            xlabel('');
        else
            xlabel('Input $V_1$','interpreter','latex')
        end
        
        if c == 1
            ylabel('Input $V_2$','interpreter','latex')
        else
            ylabel('');
        end
        
        if r == 1
            if c == 2
                title('Standard Clamping')
            elseif c==5
                title('Overclamping')
            end
        end
        
   
    end
end


exportgraphics(gcf,['Fig5.pdf'],'BackgroundColor','none');



