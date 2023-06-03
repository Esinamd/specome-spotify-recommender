import { Link } from 'react-router-dom';

const Home = () => {
  return (
    <div>
      <div className="titleHeader">
        <h1 className="Title Home">SpecoMe</h1>
        <h3>A Spotify Recommendation App</h3>
      </div>

      <div className="buttonPos homeButton">
        <Link to="/Songs1">
          <button>Recommend Songs</button>
        </Link>
        <Link to="/Artists1">
          <button>Recommend Artists</button>
        </Link>
      </div>

      <h4 className="footer">
        Recommending songs and artists from 1921 - 2020 <br />
        Developed by Esinam Dake
      </h4>
    </div>
  );
};

export default Home;
