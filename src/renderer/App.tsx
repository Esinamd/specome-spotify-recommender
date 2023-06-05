import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import './App.css';
import Home from './pages/Home';
import PageSongs1 from './pages/PageSongs1';
import PageSongs2 from './pages/PageSongs2';
import PageArtists1 from './pages/PageArtists1';
import PageArtists2 from './pages/PageArtists2';
import RecSongs from './pages/RecSongs';
import RecArtists from './pages/RecArtists';
import SpotifyConnect from './pages/SpotifyConnect';
import SpotifyReturn from './pages/SpotifyReturn';

export default function App() {
  return (
    <>
      <Router>
        <Routes>
          <Route path="/index.html" element={<Home />} />
          <Route path="/" element={<Home />} />
          <Route path="/Songs1" element={<PageSongs1 />} />
          <Route path="/Songs2" element={<PageSongs2 />} />
          <Route path="/Artists1" element={<PageArtists1 />} />
          <Route path="/Artists2" element={<PageArtists2 />} />
          <Route path="/RecSongs" element={<RecSongs />} />
          <Route path="/RecArtists" element={<RecArtists />} />
          <Route path="/SpotifyConnect" element={<SpotifyConnect />} />
          <Route path="/SpotifyReturn" element={<SpotifyReturn />} />
        </Routes>
      </Router>
    </>
  );
}
