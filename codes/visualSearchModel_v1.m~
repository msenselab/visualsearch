function visualSearchModel1()

Ns    = [8 12 16];  % Set sizes
leaks = [0 1];      % Memory leak (0: static task; 1: dynamic task)

p0  = 0.5;          % Prior probability of target presence
r   = 0.15;         % Reward for a correct answer
tw  = 1;            % Non-decision time + ITI
rho = 0;            % Reward rate (to be optimized) (Set this 0 for a single trial models)
c   = 0.001;        % Cost per time (cognitive effort etc.)

t = 0:40;           % Time steps
dt = 0.1;           % Unit time step

thresh_ap = (c+rho)*dt/r; % Threshold for "Absent" response based on "a_t * p_t+1" index


pt = @(t,N,leak) qt(t,N,leak) * p0 ./ ((qt(t,N,leak)-1)*p0+1);         % Posterior probability of target presence
at = @(t,N,leak) pt(t,N,leak) ./ Nt(t,N,leak);
ap = @(t,N,leak) at(t,N,leak) .* pt(t+1,N,leak);


rtAbs = findCrossPoint(t, Ns, leaks, thresh_ap, @(t,N,leak) ap(t,N,leak));      % RT for "Absent" trials
rtPre = findCrossPoint(t, Ns, leaks, 0.5, @(t,N,leak) qt(t,N,leak));            % Median RT for "Present" trials

%% - Show results -
figure;
fontsize = 11;

subplot(2,2,1); hold on;
plotFunctions(t, Ns, leaks, @(t,N,leak) qt(t,N,leak));
plot([0 max(t)], 0.5 * [1 1], '-', 'Color',0.7*[1 1 1]);
xlabel('t'); ylabel('q_t:=P(no target obs.|present)');
title('(A) Likelihood');
set(gca, 'FontSize',fontsize);
legend({'Static','Dynamic'});

subplot(2,2,2); hold on;
plotFunctions(t, Ns, leaks, @(t,N,leak) pt(t,N,leak));
xlabel('t'); ylabel('p_t:=P(present|no target obs.)');
title('(B) Posterior');
set(gca, 'FontSize',fontsize);

subplot(2,2,3); hold on;
plotFunctions(t, Ns, leaks, @(t,N,leak) at(t,N,leak));
xlabel('t'); ylabel('\alpha_t');
title({'(C) Probability of observing';'the target in the next time step'});
set(gca, 'FontSize',fontsize);

subplot(2,2,4); hold on;
plotFunctions(t, Ns, leaks, @(t,N,leak) ap(t,N,leak));
plot([0 max(t)], thresh_ap * [1 1], '-', 'Color',0.7*[1 1 1]);
text(max(t), thresh_ap, num2str(thresh_ap), 'Color',0.7*[1 1 1]);
xlabel('t'); ylabel('\alpha_t p_{t+1}');
title('(D) \alpha_t p_{t+1}');
set(gca, 'FontSize',fontsize);

inset(gca, [.6 .4 .4 .5]); hold on;
rtPre = rtPre ./ repmat([1 log(2)],[length(Ns) 1]);     % Median RT -> mean RT for exponential PDF
plotLines(Ns, rtAbs, leaks, 'o-');
plotLines(Ns, rtPre, leaks, '^:');
xlabel('Set size, N'); ylabel('Mean RT');
set(gca, 'FontSize',0.7*fontsize, 'XTick', Ns);
xlim(axisMinMax(Ns,0.3));



function q = qt(t, N, leak)
% Likelihood of observing no target in a "presnt" trial 
switch leak
    case 1	% Dinamic task
        q = (1-1/N).^t;
    case 0	% Static task
        q = max(0,1-t/N);
end

function Nt_ = Nt(t, N, leak)
% Number of remaining items
switch leak
    case 1  % Dinamic task 
        Nt_ = N*ones(size(t));
    case 0  % Static task
        Nt_ = max(0,N-t);
end


function tCross = findCrossPoint(t, N, leak, thresh, f)
for iN = 1:length(N)
    for iLeak = 1:length(leak)
        err = abs(f(t,N(iN),leak(iLeak)) - thresh);
        tCross(iN,iLeak) = min(t(err==min(err)));
    end
end
 
function plotFunctions(t, N, leak, f)
colLeak = {[0 0.7 0.7],[1 .1 0]};
mkrN = {'-','--',':'};
for iN = 1:length(N)
    for iLeak = 1:length(leak)
        plot(t, f(t,N(iN),leak(iLeak)), mkrN{iN}, 'Color',colLeak{iLeak});
    end
end

function plotLines(N, y, leak, mkr)
colCond = {[0 0.7 0.7],[1 .1 0]};
for iLeak = 1:length(leak)
    plot(N, y(:,iLeak),  mkr, 'Color',colCond{iLeak});
end