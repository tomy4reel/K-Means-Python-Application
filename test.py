import unittest
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class TestKMeansClustering(unittest.TestCase):
    
    def setUp(self):
        self.data = pd.DataFrame({
            'Sex': [1, 0, 0, 0, 0],
            'Marital status': [0, 0, 0, 0, 1],
            'Age': [82, 63,54, 45, 61],
            'Education': [1, 0, 0, 0, 0],
            'Income': [50000, 40000, 3000, 20000, 10000],
            'Settlement size': [0, 0, 2, 1, 1],
            
        })
        self.features = self.data[["Marital status","Age", "Education"]]
        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(self.features)
        self.kmeans = KMeans(n_clusters=2, random_state=42)
        self.data['Cluster'] = self.kmeans.fit_predict(self.scaled_features)
    
    def test_cluster_assignment(self):
        # Check if clustering is done correctly
        self.assertTrue('Cluster' in self.data.columns)
        self.assertEqual(len(self.data['Cluster'].unique()), 2)
    
    def test_silhouette_score(self):
        # Test silhouette score
        from sklearn.metrics import silhouette_score
        silhouette_avg = silhouette_score(self.scaled_features, self.data['Cluster'])
        self.assertGreater(silhouette_avg, 0.5)  # Example threshold

if __name__ == '__main__':
    unittest.main()
