from flask import Flask, request, jsonify
from model.gnn_utils import GNNUtils
from torch.multiprocessing import Pool, Process, set_start_method


app = Flask(__name__)
try:
    set_start_method('spawn')
except RuntimeError:
    pass
gnn_utils = GNNUtils()


@app.route('/recommend', methods=['GET'])
def recommend():
    user = request.args.get('user_id')
    quantity = int(request.args.get('quantity'))
    rec = gnn_utils.recommend_for_user(user, quantity)
    print(rec)    
    return jsonify({'recommendations': rec })


@app.route('/add-new-user', methods=['POST'])
def add_new_user():
    try:
        data = request.json 
        
        print("Before =>>", GNNUtils.graph)
        result = gnn_utils.add_new_user_feat(data['userId'], data['gender'])
        print("After =>>", GNNUtils.graph)
        return jsonify({'success': result })
    except Exception as e:
        print(e.with_traceback())
        return jsonify({'success': False })

@app.route('/add-new-rating', methods=['POST'])
def add_new_rating():
    try: 
        data = request.json 
        result = gnn_utils.add_new_edge(data['userId'], data['productId'], data['rating'])
        print("After", GNNUtils.graph)
        return jsonify({'success': result })
    except Exception as e:
        print(e.with_traceback())
        return jsonify({'success': False })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)