function finehist(sigma)
%FINEHIST Distribution of decision variables for fine-grained model.
%   Input: SIGMA    Noise for each stimulus (assumed the same for all stimuli)

N = [8,12,16];  % Number of stimuli
Ns = 3e4;       % Number of samples for computation of the histogram

ymax = 0;

for iPanel = 1:numel(N)
    subplot(1,numel(N),iPanel);
    plotpanel(sigma,N(iPanel),Ns);
    % Get current panel maximum y-axis
    ylim = get(gca,'ylim');
    ymax = max(ymax,ylim(2));
end

% Uniform y axis
for iPanel = 1:numel(N)
    subplot(1,numel(N),iPanel); 
    set(gca,'ylim',[0 ymax]);
end

set(gcf,'Color','w');   % Set bkg color to white

end

%--------------------------------------------------------------------------
function [d0,d1] = plotpanel(sigma,N,Ns)
%PLOTPANEL Plot histogram for a given value of SIGMA and number of stimuli N.

fontsize = 16;
axesfontsize = 12;

eta = sigma*randn(Ns,N);    % Random numbers
sigma2 = sigma^2;

K = -0.5/sigma2 - log(N);   % Constant

d0 = K + log(sum(exp(eta/sigma2),2));
d1 = K + log(exp((eta(:,1)+1)/sigma2) + sum(exp(eta(:,2:end)/sigma2),2));

[~,pdf0,xx0] = kde(d0);
[~,pdf1,xx1] = kde(d1);

plot(xx0,pdf0,'-g','LineWidth',2); hold on;
plot(xx1,pdf1,'-b','LineWidth',2);

title(['\sigma = ' num2str(sigma) ', N = ' num2str(N)],'FontSize',fontsize);

% Prettify plot
set(gca,'TickDir','out','FontSize',axesfontsize);
box off;

xlabel('Decision variable','FontSize',fontsize);
ylabel('Probability density','FontSize',fontsize);

leg0 = ['Target absent (mean = ' num2str(mean(d0),'%.2f') ', SD = ' num2str(std(d0),'%.2f') ')'];
leg1 = ['Target present (mean = ' num2str(mean(d1),'%.2f') ', SD = ' num2str(std(d1),'%.2f') ')'];

h = legend(leg0,leg1);
set(h,'Box','off','Location','NorthEast','FontSize',axesfontsize);

end