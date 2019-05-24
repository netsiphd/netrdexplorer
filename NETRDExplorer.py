from flask import Flask, jsonify, request, send_file, render_template
import networkx as nx
import pandas as pd
from netrd import dynamics
from flask_restful import Resource, Api
import numpy as np


import netrd.reconstruction as recon
import netrd.distance as distt

reconOptions = ['Random','Correlation Matrix','Regularized Correlation Matrix','Partial Correlation Matrix',
           'Free Energy Minimization','Naive Mean Field','Thouless Anderson Palmer','Exact Mean Field',
           'Maxium Likelihood Estimation','Covergent Cross Mapping','Mutual Information Matrix','OU Inference',
           'Graphical Lasso','Marchenko Pastur','Time Granger Causality', 'Correlation Spanning Tree']

recons = {'Random': recon.RandomReconstructor(),
          'Correlation Matrix': recon.CorrelationMatrix(),
          'Free Energy Minimization': recon.FreeEnergyMinimization(),
          'Correlation Matrix': recon.CorrelationMatrix(),
          'Partial Correlation Matrix': recon.PartialCorrelationMatrix(),
          'Partial Correlation Influence': recon.PartialCorrelationInfluence(),
          'Naive Mean Field': recon.MeanField(),
          'Thouless Anderson Palmer': recon.ThoulessAndersonPalmer(),
          'Exact Mean Field': recon.MeanField(),
          'Maxium Likelihood Estimation': recon.MaximumLikelihoodEstimation(),
          'Covergent Cross Mapping':  recon.ConvergentCrossMapping(),
          'Mutual Information Matrix':  recon.MutualInformationMatrix(),
          'OU Inference': recon.OUInference(),
          'Graphical Lasso': recon.GraphicalLasso(),
          'Marchenko Pastur': recon.MarchenkoPastur(),
          'Time Granger Causality': recon.TimeGrangerCausality(),
          'Correlation Spanning Tree': recon.CorrelationSpanningTree()}

distanceOptions = ['Jaccard','Hamming','Frobenius','Portrait Divergence',
                   'Ipsen Mikhailov','Hamming Ipsen Mikhailov','Resistance Perturbation','Net Simile', 'Laplacian Spectral']

dist = {'Jaccard': distt.JaccardDistance(),
        'Hamming': distt.Hamming(),
        'Frobenius': distt.Frobenius(),
        'Portrait Divergence': distt.PortraitDivergence(),
        'Ipsen Mikhailov': distt.IpsenMikhailov(),
        'Hamming Ipsen Mikhailov': distt.HammingIpsenMikhailov(),
        'Resistance Perturbation': distt.ResistancePerturbation(),
        'Net Simile': distt.NetSimile(),
        'Polynomial Dissimilarity': distt.PolynomialDissimilarity(),
        'Onion Divergence': distt.OnionDivergence(), 
        'Quantum Spectral JSD': distt.QuantumJSD(),
        'Degree Divergence': distt.DegreeDivergence(), 
        'Net LSD': distt.NetLSD(),
        'Communicability Sequence': distt.CommunicabilityJSD(),
        'Non Backtracking Distance': distt.NonBacktrackingSpectral(),
        'Delta Con': distt.DeltaCon()}

netGen = {'Ring of Cliques': nx.ring_of_cliques,
          'Barabasi Albert Graph': nx.barabasi_albert_graph,
          'Erdos Renyi Graph': nx.erdos_renyi_graph,
          'Random Geometric Graph' :nx.random_geometric_graph}

timeGen = {'Sherrington Kirkpatrick Ising': dynamics.SherringtonKirkpatrickIsing,
           'Branching Model': dynamics.BranchingModel,
           'Voter Model': dynamics.VoterModel,
           'Single Unbiased Random Walker': dynamics.SingleUnbiasedRandomWalker,
           'Ising Glauber': dynamics.IsingGlauber}

app = Flask(__name__)

api = Api(app)

class NetworkGenerator(Resource):

    def put(self):
        global reconCache
        reconCache = {}
        gen = request.args.get('gen', default = 'Ring of Cliques', type = str)
        cliq = request.args.get('cliq', default = "1", type = str)
        size = request.args.get('size', default = "1", type = str)
        if gen == 'Erdos Renyi Graph' or gen == 'Random Geometric Graph':
            cliq = int(cliq)
            size = float(size)
        else:
            cliq = int(cliq)
            size = int(size)
        G = netGen[gen](cliq, size)
        while not nx.is_connected(G):
            G = netGen[gen](cliq, size)
        network = nx.node_link_data(G)
        return jsonify(network)



class TimeSeries(Resource):

    def put(self):
        try:
            gen = request.args.get('gen', default = 'Sherrington Kirkpatrick Ising', type = str)
            network = request.get_json()
            G = nx.node_link_graph(network)
            len = request.args.get('len', default = 2000, type = int)
            dynamics = timeGen[gen]()
            ts = dynamics.simulate(G, len)
            df = pd.DataFrame(ts)
            #plt.imshow(ts,aspect='auto',cmap='Reds')
            #plt.savefig('static/hi2.png')
            return jsonify(df.to_json(orient='records'))
            #return send_file('hi2.png', mimetype='image/png')
        except Exception as e:
                return str(e)

class Metrics(Resource):

    def put(self):
        network = request.get_json()
        G = nx.node_link_graph(network)
        meanDegree = np.mean(list(dict(nx.degree(G)).values()))
        return meanDegree

class Scale(Resource):

    def put(self):
       
        try:
            meth = request.args.get('dist', default = 'Jaccard', type = str)
            avgDegree = request.args.get('avg', default = '0', type = str)
            avgDegree = float(avgDegree)
            reconstructor = recons['Random']
            distance = dist[meth]
        except Exception as e:
                return str(e)
        try:
            x = request.get_json()
            df = pd.DataFrame(x['ts'])
            ts = np.array(df)
            G = reconstructor.fit(ts, threshold_type='degree', avg_k=avgDegree, binary=True)
            res = distance.dist(G,nx.node_link_graph(x['G']))
            return str(res)
        except Exception as e:
                return str(e)

class Reconstructor(Resource):

    def put(self, reconID):

        try:
            recon = request.args.get('recon', default = 'Random', type = str)
            avgDegree = request.args.get('avg', default = '0', type = str)
            avgDegree = float(avgDegree)
            reconstructor = recons[recon]
        except Exception as e:
                return str(e, 'hi')

        if (recon == 'Free Energy Minimization'):
            return 'hi'

        try:
            x = request.get_json()
            df = pd.DataFrame(x)
            ts = np.array(df)
            if (recon == 'Naive Mean Field'):
                G = reconstructor.fit(ts, exact=False, threshold_type='degree', avg_k=avgDegree, binary=True)
            else:
                G = reconstructor.fit(ts, threshold_type='degree', avg_k=avgDegree, binary=True)
            return jsonify(nx.node_link_data(G))
        except Exception as e:
                return str(e, ts.shape)

class Distance(Resource):

    
    def put(self):
        try:
            graphs = request.get_json()
            meth = request.args.get('dist', default = 'Jaccard', type = str)
            g1 = request.args.get('g1', default = 'N/A', type = str)
            g2 = request.args.get('g2', default = 'N/A', type = str)
            g3 = request.args.get('g3', default = 'N/A', type = str)
            names = ["Ground Truth",g1,g2,g3]
            distance = dist[meth]
            df = pd.DataFrame()
            graphs = [nx.node_link_graph(graphs['g0']),nx.node_link_graph(graphs['g1']),
                      nx.node_link_graph(graphs['g2']),nx.node_link_graph(graphs['g3'])]
            for i in range(4):
                for j in range(4):
                    #print(i,j,meth.dist(graphs[i],graphs[j]))
                    df = df.append([[names[i],names[j],distance.dist(graphs[i],graphs[j])]], ignore_index=True)
            df.columns = ['group', 'variable', 'value']
            return jsonify(df.to_json(orient='records'))
        except Exception as e:
                return str(e)



api.add_resource(NetworkGenerator, '/network')
api.add_resource(TimeSeries, '/time')
api.add_resource(Reconstructor, '/recon/<string:reconID>')
api.add_resource(Distance, '/distance')
api.add_resource(Metrics, '/metrics')
api.add_resource(Scale, '/scale')


@app.route('/')
def webpage():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
