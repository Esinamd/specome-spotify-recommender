import { Link, useNavigate } from 'react-router-dom';

const PageSongs1 = () => {
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

      <h2 className="subheadPos">Recommend songs based on...</h2>

      <div className="buttonPos">
        <Link to="/Songs2">
          <button className="pagebutton">One Song</button>
        </Link>
        <Link to="/SpotifyConnect" state={{ type: 'songs' }}>
          <button className="pagebutton">Song Library</button>
        </Link>
      </div>
    </div>
  );
};
export default PageSongs1;
