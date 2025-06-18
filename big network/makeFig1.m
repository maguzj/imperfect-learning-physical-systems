%% Classification cycling with entire network
nodemult = 0.4532;  % conversion from range to volts
figure(2412101)
clf
bigaxs = [];
N = [6,5];
shapes = {'o','s'};
for k = 1:prod(N)
    bigaxs(k) = subplot(N(1),N(2),k);
end
k0 = 1;

ticksize = 0.04;
            yL = {[10^-5.2,1.3*10^-2],[10^-2.5,10^1.45]};


for whichrun = [1,2]
    
    twodatapoints = false;
    shiftval = -2;
    switch whichrun
        
        case 1
            name = 'taskcycle_I_06';
            % pickvals = [14,17,20]+shiftval;
            pickvals = [14,20]+shiftval;
            endvec = -20:-1;
            lineshift = 0;
            traceYL = [-.001,0.007];
            traceDelta = 0.003;
            ABlocs = nodemult*[.6,.32;.4,.7];
            
            taskYL = [0,.5];
            
        case 2
            name = 'memorycyclewhole_A_06';
            % pickvals = 4+(14:3:20)+shiftval;
            pickvals = 4+(14:6:20)+shiftval;
            endvec = -40:-1;
            lineshift = 0;
            traceYL = [-.005,0.021];
            traceDelta = 0.01;
            ABlocs = nodemult*[.40,-.137;.6,.048];
            
            
            taskYL = [-.1,.1];
            
    end
    xL0 = [5*10^-4,30];
    xL1 = [7*10^-4,30];
    
    loadEG = false;
    try
        EGs{whichrun}.Name
        if not(strcmp(EGs{whichrun}.Name,name))
            loadEG = true;
        end
    catch
        loadEG = true;
    end
    
    if loadEG
        EG = ExperimentGroup(name,endvec,[],twodatapoints);
        EGs{whichrun} = EG;
    else
        EG = EGs{whichrun};
    end
    
    
    %%
    
    axs = [];
    for k = 1:15
        axs(k) = bigaxs(k0);
        k0 = k0+1;
    end
    
    axs(14:15) = bigaxs(14:15);
    
    
    linespec = {shapes{whichrun},'LineWidth',1.5,'markersize',8,'capsize',0};
    
    colors = repmat(colororder,4,1);
    colors2 = {[.8,.2,.2],[.2,.2,.8]};
    colors2 = {1.3*[141,27,17]/255, 1.3*[10,30,90]/255};
    colors3 = {};
    for k = 1:length(colors2)
        colors3{k} = colors2{k}*0.7+0.3;
    end
    
    
    
    
    
    idxvals = unique(-EG.TRAINIDX);
    TAU0s = idxvals(pickvals);
    ERRVAL = EG.DELTAERRTRAIN;
    wanthinge = and(EG.CLA>1,EG.UPD>0);
    ERRVAL(wanthinge) = EG.DELTAHINGETRAIN(wanthinge);
    vals = {ERRVAL,EG.DELTAG};
    tags = {'Combined Error $\overline{E}$ (V$^2$)','Cycle Span $D$ (V)'};
    
    
    X = -EG.TRAINIDX.*EG.DUP.*EG.ALF/10^6;
    try
        
        X(EG.GRU>1) = X(EG.GRU>1).*EG.TRA(EG.GRU>1)./EG.GRU(EG.GRU>1);
    catch
    end
    
    X0s = TAU0s*0;
    timestrs = {};
    for j = 1:length(TAU0s)
        X0s(j) = unique(2*X(-EG.TRAINIDX==TAU0s(j)));
        
        val = round(X0s(j),2-ceil(log10(X0s(j))));
        if val<.1
            val = val*1000;
            timestr = 'ms';
        else
            timestr = 's';
        end
        
        ordr = floor(log10(val));
        val = round(val/10^ordr)*10^ordr;
        timestr = [num2str(val),timestr];
        timestrs{j} = timestr;
        
        
        
        EXP = EG.loadExperiment(find(EG.TRAINIDX==-TAU0s(j),1));
        [a,b] = EXP.nameInputs;
        sidx = find(strcmp(b,'$V_1$'));
        
        
        if EXP.GRU>1
            TAU = TAU0s(j)*EXP.TRA/EXP.GRU;
        else
            TAU = TAU0s(j);
        end
        idxs = find(rem(EXP.DOTEST,TAU)==1);
        mes = [];
        for t = 1:length(idxs)
            
            idx = idxs(t);
            
            cycle = find(and(EXP.DOTEST>=EXP.DOTEST(idx),EXP.DOTEST<=(EXP.DOTEST(idx)+2.1*TAU)));
            if length(cycle)>=length(mes)
                
                mes = cycle;
                startidx = idx;
                mididx = cycle(find(EXP.DOTEST(cycle)>=EXP.DOTEST(idx)+TAU/2,1));
            end
            
        end
        
        if EXP.CLA>1
            
            [~,~,~,~,~,hingeerr,errorTRAIN] = EXP.FoundClassIDs();
            errorTRAIN = errorTRAIN(mes,:)*nodemult;
        else
            outputs = permute(diff(EXP.TrainMeasurements,1,1),[3,2,1]);
            
            errorTRAIN = outputs;
            for t = 1:EXP.TRA
                errorTRAIN(:,t) = nodemult*(errorTRAIN(:,t)-diff(EXP.TRAIN(EXP.SOR+1:end,t),1,1));
                
            end
            outputs = outputs * nodemult;
            errorTRAIN = errorTRAIN(mes,:);
        end
        
        
        times = cumsum(EXP.LearnTimes)*10^-6;
        times = times(mes)-times(mes(1));
        times = times/X0s(j);
        
        
        if EXP.GRU>1
            testhinge = mean(errorTRAIN(:,EXP.TRAINGROUP==(0)).^2,2);
        else
            testhinge = mean(errorTRAIN(:,jj).^2,2);
            
        end
        
        if testhinge(find(times>0,1)) > testhinge(find(times>0.5,1))
            shifton = true;
        else
            shifton = false;
        end
        
        if EXP.CLA==1
            shifton = not(shifton);
        end
        if shifton
            dummy =  mididx;
            mididx = startidx;
            startidx = dummy;
        end
        
        
        
        axes(axs(5+2*j-1))
                 set(gca,'xaxislocation','top')

        if EXP.CLA>1
            nonlinearClassificationMap2(EXP,mididx,colors3,false)
                  %   set(gca,'xaxislocation','top')

            %   set(gca,'xtick',[0,200],'ytick',[0,200],'xticklabel',{0,'V_{max}'});
          % set(gca,'ytick',[10,200-5],'xtick',[10,200-5],'yticklabel',{'V_-','V_+'},'xticklabel',{'V_-','V_+'})
           if j == 1
             ylabel('$V_2$','interpreter','latex')
           else
               ylabel('')
               set(gca,'ytick',[])

           end
           xlabel('$V_1$','interpreter','latex')%,'position',[100,-30])
            title('')
            %             xx = 35;
            %             rectangle('Position',[-1,201-2*xx,2*xx,2*xx],'FaceColor','k','EdgeColor','none')
            %
            %             text(15,205,'$\tau$','verticalalignment','top','horizontalalignment','left',...
            %                 'FontWeight','bold','FontSize',17,'Color','w','Interpreter','latex',...
            %                 'backgroundcolor','none')
            x = get(gca,'children');
            for kk = 1:length(x)
                try
                    if sum(abs(x(kk).MarkerFaceColor - [0.6425 0.3171 0.2885]))<0.05
                        x(kk).MarkerFaceColor = [0.6425 0.3171 0.2885]*0.5+0.5;
                        x(kk).MarkerSize = 4;
                    elseif sum(abs(x(kk).MarkerFaceColor -   [0.2685 0.3256 0.4969]))<0.05
                        x(kk).MarkerSize = 6;
                        x(kk).Color = 'k';
                        x(kk).LineWidth = 0.5;
                    end
                catch
                end
            end
            
           % if whichrun==1
           % title(['\tau = ',timestr],'Position',[220,215])
           % else
           title('')
           %end
           
         
           if j == 2 && whichrun==2
                %xlabel('Task \alpha','color',colors2{2})
                title('$t = \tau/2$','color','k','interpreter','latex')
                
            elseif j== 2 && whichrun==1
                cb = colorbar('orientation','vertical','location','eastoutside','TickLabelInterpreter','latex');
                cb.Ticks = [-1,0,1]*.063;
                cb.TickLabels = {'$L_-$','0','$L_+$'};
                % ylabel(cb,'Output','FontSize',12)
            end
        else
            
            for jj = 1:2
                color = colors2{jj};
                plot(EXP.TRAIN(sidx,jj)*nodemult,diff(EXP.TRAIN(EXP.SOR+1:end,jj),1,1)*nodemult,...
                    shapes{whichrun},'MarkerSize',12,'Color',(color*(2-(jj==1))+(jj==1))*0.5,...
                    'MarkerFaceColor','none','LineWidth',3)
                
                
                hold on
            end
            ylim([-.1,.1])
            
            
            plot(EXP.TRAIN(sidx,1:2)*nodemult,outputs(mididx,:),'-o','color','k','LineWidth',3,'MarkerSize',3)
            hold on

            
            ylabel('')
            set(gca,'ytick',-.1:.1:.1,'xtick',[0,nodemult],'xticklabel',{0,'V_{+}'});
            h = plot([0,nodemult]+[-1,1],[0,0],'--','color',[1,1,1]*0.5,'LineWidth',1);
            uistack(h,'bottom')
            xlim([0,nodemult]+[-2,2]*nodemult*.1)
        xlabel('$V_1$','interpreter','latex')
                set(gca,'xticklabel',[])

            if j == 1
                ylabel('O (V)')
            else
                set(gca,'yticklabel',[]);             
            end
            
%             if j == 2
%                 title('$t = \tau/2$','interpreter','latex')
%                % set(gca,'xticklabel',[])
%             end
            
        end
                         set(gca,'xaxislocation','top')

        axes(axs(5+2*j))

        if EXP.CLA>1
            
            nonlinearClassificationMap2(EXP,startidx,colors3,false)
           %  set(gca,'ytick',[],'xtick',[10,200-5],'xticklabel',{'V_-','V_+'})
       %  set(gca,'xaxislocation','top')
               ylabel('')
           xlabel('$V_1$','interpreter','latex')%,'position',[100,-30])
            title('')
            %              xx = 35;
            %             rectangle('Position',[201-2*xx,-1,2*xx,2*xx],'FaceColor','k','EdgeColor','none')
            %
            %             text(205,15,'$\tau/2$','verticalalignment','bottom','horizontalalignment','right',...
            %                 'FontWeight','bold','FontSize',17,'Color','w','Interpreter','latex',...
            %                 'backgroundcolor','none')
            x = get(gca,'children');
            for kk = 1:length(x)
                try
                    if sum(abs(x(kk).MarkerFaceColor - [0.2685 0.3256 0.4969]))<0.05
                        x(kk).MarkerFaceColor = [0.2685 0.3256 0.4969]*0.5+0.5;
                        x(kk).MarkerSize = 4;
                    elseif sum(abs(x(kk).MarkerFaceColor -   [0.6425 0.3171 0.2885]))<0.05
                        x(kk).MarkerSize = 6;
                        x(kk).Color = 'k';
                        x(kk).LineWidth = 0.5;
                    end
                catch
                end
            end
            
            
%             if j == 2
%                 % xlabel('Task \beta','color',colors2{1})
%                 xlabel('$t = \tau$','color','k','interpreter','latex')
%             else
            %    if j == 2
          %      xlabel('Output')
          %  end
        else
            
            for jj = 1:2
                color = colors2{jj};
                plot(EXP.TRAIN(sidx,jj)*nodemult,diff(EXP.TRAIN(EXP.SOR+1:end,jj),1,1)*nodemult,...
                    shapes{whichrun},'MarkerSize',12,...
                    'Color',(color*(2-(jj==2))+(jj==2))*0.5,...
                    'MarkerFaceColor','none','LineWidth',3)
                
                
                hold on
            end
            
            % plot(EXP.TRAIN(sidx,1:2)*nodemult,outputs(startidx,:),'-','color',colors2{1},'LineWidth',2)
            plot(EXP.TRAIN(sidx,1:2)*nodemult,outputs(startidx,:),'-o','color','k','LineWidth',3,'MarkerSize',3)
            
            h = plot([0,nodemult]+[-1,1],[0,0],'--','color',[1,1,1]*0.5,'LineWidth',1);
            uistack(h,'bottom')
            xlim([0,nodemult]+[-2,2]*nodemult*.1)
            if whichrun == 3
                ylim([0,.5])
            else
                ylim([-.1,.1])
            end
            set(gca,'ytick',-.1:.1:.1,'xtick',[0,nodemult],'xticklabel',{0,'V_{+}'});
            
            set(gca,'yticklabel',[])
            ylabel('')
            
             xlabel('$V_1$','interpreter','latex')
              set(gca,'xticklabel',[])

           
%             if j == 2
%                 title('$t = \tau$','interpreter','latex')
%                 set(gca,'xticklabel',[])
%             end
            ax = gca;
            ax.TickLength = [1,1]*ticksize;
        end
                         set(gca,'xaxislocation','top')

        
        
        axes(axs(1+j))
        
        for jj = 1:2
            if EXP.GRU>1
                hinge = mean(errorTRAIN(:,EXP.TRAINGROUP==(jj-1)).^2,2);
            else
                hinge = mean(errorTRAIN(:,jj).^2,2);
                
            end
            xvals = [times-times(end), times, times+times(end)];
            yvals = [hinge; hinge; hinge];
            
            
            if shifton
                xvals = xvals - 0.5;
            end
            plot(xvals,yvals,'-','LineWidth',2,'color',colors2{jj})
            hold on
        end
       % xlabel('')
        tickvec = -10*traceDelta:traceDelta:10*traceDelta;
        
        if j ==1
            % if EXP.CLA>1
            ylabel('E (V^2)')
            %  else
            % ylabel('Error (V)')
            % end
            set(gca,'ytick',tickvec,'yticklabel',tickvec);
            ax = gca;
            str = ['\tau = ',timestr,''];
        else
            ylabel('')
            set(gca,'ytick',tickvec,'yticklabel',[])
            str = timestr;
        end
        
        
        
        if j == 1
            text(0,traceYL(2)*0.6,'E_\alpha','color',colors2{2},...
                'FontWeight','bold','FontSize',17);
            
            text(.3,traceYL(2)*0.6,'E_\beta','color',colors2{1},...
                'FontWeight','bold','FontSize',17);
            
        end
        
        v = -traceYL(2)/7;
        
        
        % text(1,traceYL(2),str,'HorizontalAlignment','right','VerticalAlignment','top','FontSize',13)
        
        set(gca,'xtick',[0,.5,1],'xticklabel',{'0','','\tau'},'xaxislocation','bottom')
        hh = xlabel('Time (s)');
        if whichrun==1
            hh.Position(2) = -.0017;
        else
            hh.Position(2) = -.0055;
        end
        hh.Position(1) = .5;
        xL = [-.1,1.1];
        % ylim([v,diff(traceYL)])
        ylim([v,traceYL(2)]);
        xlim(xL)
        %         if EXP.CLA<2
        %             h = plot(xL,[0,0]-traceYL(1),'--','color',[1,1,1]*0.5,'LineWidth',1);
        %             uistack(h,'bottom')
        %         end
        
        %         area([-.5,.5],[1,1]*v,'facecolor',colors2{2}*0.5+0.5,'edgecolor','none')
        %         area([.5,1.5],[1,1]*v,'facecolor',colors2{1}*0.5+0.5,'edgecolor','none')
        set(gca,'layer','top')
        ax = gca;
        ax.TickLength = [1,1]*ticksize;
       
    end
    
    for s = 1:2
        [x,y,err,cat] = makeErrorBar(2*X,vals{s},EG.TRAINSET==1,[],true,false);
%         want = and(x>xL1(1), x< xL1(2));
%         x = x(want);
%         y = y(want);
%         err = err(want);
%         want
%         disp('clipping values')
%         

        axes(axs(s+13)); %#ok<LAXES>
        % fill = colors(3,:);
        % if whichrun == 2
        fill = 'w';
        % end
        errorbar(x,y,err,linespec{:},'color',colors(3,:)*(1-.3*(whichrun==2)),'markerfacecolor',fill)
        hold on
        
        set(gca,'xscale','log','yscale','log','xtick',logspace(-10,10,21),'ytick',logspace(-10,10,21))
        xlabel('Period \tau (s)')
        
        
        ylim(yL{s});
        xlim(xL0);
        
        ylabel(tags{s},'interpreter','latex')
        if s == 1
            if EXP.GRU>1
                set(gca,'yscale','lin','ytick',10^-3*(0:5))
            else
                set(gca,'yscale','log');%'ytick',.25*10^-2*(0:5))
                
            end
        else
            plot(10.^(-.5+[-2.3,0]+lineshift),10.^((1+[-2.3,0]+lineshift)),'k-','LineWidth',2)
        end
        
        %         for j = 1:length(X0s)
        %             idx = find(x==X0s(j));
        %             color = 'k';%colors(j*2,:)*0.9+0.1;
        %             plot(x(idx),y(idx),linespec{:},'MarkerSize',12,'MarkerFaceColor','none','Color',color)
        %         end
    end
    
    
    shftA = 0.01;
    %%
    set(gcf,'color','none')
    for k = 1:length(axs)
        axes(axs(k))
        makePlotPrettyNow(12)
        if isempty(get(gca,'Children'))
            axis off
        end
    end
    
    traceW = .12;
    YS = .7;
    S = [1,YS,1,YS];
    S0 = .04;
    F = [0,.25+.30*(1-whichrun),0,0];
    set(gcf,'units','pixels','position',[10,100,1250,600/YS])
    
    if EXP.CLA>1
        set(axs(1),'innerposition',[.03,.7,.13,.25].*S+F)
        
    else
        set(axs(1),'innerposition',[.03,.7,.13,.25].*S+F)
    end
    for k = 0:3
        set(axs(k+2),'innerposition',[.2195+(traceW+.01)*k-shftA*(k<1),.79-.12+S0,traceW+.003,.12].*S+F)
    end
    
    
    for k = 0:7
        vec = [.218+(.005 + traceW/2)*k-0.001*rem(k,2)-shftA*(k<2),.66+.2+S0-.03,(traceW)/2,.12].*S+F;
                    set(axs(k+6),'innerposition',vec)

        if EXP.CLA>1
            if k == 2
             %   cb.Position = [vec(1)+.002,vec(2)-.028,vec(3)-.004,.015];
                cb.Position = [vec(1)+vec(3)*2+.01,vec(2)+.002,.015,vec(4)-.004];
            end
            
        else
            %             if rem(k,2)==1
            %                 continue
            %             end
            %set(axs(k+6),'innerposition',[.22+(traceW+0.01)*k/2,.66+.2,traceW,.12].*S+F)
            
        end
    end
    
    
    
    for k = 0:1
        set(axs(k+14),'innerposition',[.62,1.06-.35*k+S0,.15*1.215,.28*1.215].*S+F)
        axes(axs(k+14));
        if whichrun == 2
            if k == 1
                text(5*10^-3,8/16,'D \propto \tau','FontSize',14,'rotation',45)
                hold on
                for jj = 1:2
                   h = plot(X0s(jj)*[1,1],yL{2},'-','Color',[1,1,1]*.8,'LineWidth',3);
                    uistack(h,'bottom');
                   hold on 
                end
                set(gca,'layer', 'top')
            else
                
                
                hold on
                
                text(10^-3.2,.5*10^-3,'Classification','FontWeight','Normal','color',colors(3,:),'fontsize',14)
                text(10^-3.2,.16*10^-3,'Regression','FontWeight','Normal','color',colors(3,:)*.7,'fontsize',14)
                
                 for jj = 1:2
                   h = plot(X0s(jj)*[1,1],yL{1},'-','Color',[1,1,1]*.8,'LineWidth',3);
                    uistack(h,'bottom');
                   hold on 
                end
                set(gca,'layer', 'top')
                
            end
        end
        
    end
    
    %pause
    
end

grayrect = false;
if grayrect
axes(bigaxs(end));
g = gca;
k = 1;
%plot(0,0,'-');

q = .845;
set(g,'color',[1,1,1]*0.9,'innerposition',[.2125+(traceW+.013)*k,1.46-q,traceW+.01,q].*S+F)

uistack(g,'bottom')

set(get(g, 'XAxis'), 'Visible', 'off');
set(get(g, 'YAxis'), 'Visible', 'off');
end

exportgraphics(gcf,['MemoryFig1.pdf'], 'ContentType', 'vector','BackgroundColor','none');

