import { Link, useNavigate } from 'react-router-dom';

const PageArtists1 = () => {
  const navigate = useNavigate();

  return (
    <div>
      <div>
        <button className="back" onClick={() => navigate(-1)}>
          Back
        </button>
      </div>

      <div className="menuMain titleHeader">
        <h1 className="Title">SpecoMe</h1>
        <h3>A Spotify Recommendation App</h3>
      </div>

      <h2 className="subheadPos">Recommend artists based on...</h2>

      <div className="buttonPos">
        <Link to="/Artists2">
          <button className="pagebutton">One Artist</button>
        </Link>
        <Link to="/SpotifyConnect" state={{ type: 'artists' }}>
          <button className="pagebutton">Followed Artists</button>
        </Link>
      </div>
    </div>
  );
};

export default PageArtists1;
