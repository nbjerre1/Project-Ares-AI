using Godot;
using System;
using System.Linq;
using System.Collections.Generic;

public partial class Mushroom : CharacterBody2D
{
    private CharacterBody2D player;
    private HP _health;
    private bool playerInRange = false;
    private float damageCooldown = 1.0f; // seconds between hits
    private float damageTimer = 0f;

    private NavigationAgent2D navAgent;

   
    

    [Export] public float Speed = 300f;
    [Export] public int Damage { get; set; } = 10;
    [Export] public float Pushdistance = 3f;
    [Export] public PackedScene XPdropScene { get; set; }
    [Export] public PackedScene Gold { get; set; }

    private AnimatedSprite2D animatedSprite;

    public override void _Ready()
    {
        player = GetNode<CharacterBody2D>("../Player");
        

        //hp logik
        _health = GetNode<HP>("Health");
        _health.Died += OnEnemyDied;
        _health.HealthChanged += OnHealthChanged;

        // Connect Area2D signal
        var area = GetNode<Area2D>("Damage Area");
        area.BodyEntered += OnAreaBodyEntered;
        area.BodyExited += OnAreaBodyExited;

        // NavigationAgent2D setup
        navAgent = GetNode<NavigationAgent2D>("NavigationAgent2D");
        navAgent.TargetPosition = player.GlobalPosition;
        navAgent.Radius = 50f;
        MakePath();
        animatedSprite = GetNode<AnimatedSprite2D>("AnimatedSprite2D");
        animatedSprite.Play("Walk");



    }

    public override void _PhysicsProcess(double delta)
    {        
        if (player == null || navAgent == null)
            return;

        // for ai movement directly to player
        //var direction = GlobalPosition.DirectionTo(player.GlobalPosition);
        //Velocity = direction * 300;
        //MoveAndSlide();
        //if (IsInstanceValid(player) && this.GlobalPosition.DistanceTo(player.GlobalPosition) < 20) // Adjust distance as needed
        //{
        //    GD.Print("Mob is close to player, dealing damage!");
        //    player.TakeDamage(1); // Or call a method like player.TakeDamage()
        //    if (GlobalScript.PlayerHealth <= 0)
        //    {
        //        // Game over logic here
        //        GD.Print("Game Over!");
        //        GetTree().ReloadCurrentScene(); // Reloads the current scene
        //    }
        //}


        // Move along the path
        Vector2 nextPathPos = navAgent.GetNextPathPosition();
        Vector2 direction = (nextPathPos - GlobalPosition).Normalized();
        Velocity = direction * Speed;
        MoveAndSlide();

        if (animatedSprite != null)
        {
            if (Mathf.Abs(direction.X) > 0.01f) // Only flip if moving horizontally
                animatedSprite.FlipH = direction.X > 0;
        }


        // After moving, check for collision with player and apply a push
        for (int i = 0; i < GetSlideCollisionCount(); i++)
        {
            var collision = GetSlideCollision(i);
            if (collision.GetCollider() == player)
            {
                // Calculate a push direction away from the player
                Vector2 pushDir = (GlobalPosition - player.GlobalPosition).Normalized();
                // Apply a small push (tweak the multiplier as needed)
                GlobalPosition += pushDir * Pushdistance;
            }
        }
              
        
        
        // Damage logic
        if (playerInRange)
        {
            damageTimer -= (float)delta;
            if (damageTimer <= 0f)
            {
                var healthNode = player.GetNodeOrNull<HP>("Health");
                if (healthNode != null)
                {
                    healthNode.TakeDamage(Damage);
                }
                damageTimer = damageCooldown;
            }
        }
    }
    private void OnEnemyDied()
    {
       
                var xpdrop = (Node2D)XPdropScene.Instantiate();
                xpdrop.GlobalPosition = GlobalPosition;
                // calldeferred to ensure it runs after current frame
                GetParent().CallDeferred("add_child", xpdrop);
            
       
        QueueFree();
        // Optionally: emit a signal for score increase
    }
    private void OnHealthChanged(int newHp)
    {
        // Example: update health bar UI, ingen ui endnu !!!!
        var healthBar = GetNode<ProgressBar>("HealthBar");
        healthBar.Value = newHp;
        healthBar.Visible = newHp < _health.MaxHealth;
    }
    private void OnAreaBodyEntered(Node body)
    {
        if (body is CharacterBody2D && body.Name == "Player")
        {
            playerInRange = true;
            damageTimer = 0f; // So damage is applied immediately in _PhysicsProcess
        }
    }
    private void OnAreaBodyExited(Node body)
    {
        if (body is CharacterBody2D && body.Name == "Player")
        {
            playerInRange = false;
            damageTimer = 0f; // reset timer when player leaves
        }
    }

    private void MakePath()
    {
        
        if (player != null && navAgent != null)
        {
            
            navAgent.TargetPosition = player.GlobalPosition;
        }
    }
    private void _On_timer_timeout()
    {
        MakePath();
    }
    public Dictionary<string, object> GetState()
    {
        // This method is now always called from the main thread (see SocketServer.cs)
        return new Dictionary<string, object>
        {
            { "x", (float)GlobalPosition.X },
            { "y", (float)GlobalPosition.Y }
            // Add more fields if needed, e.g. { "health", health }
        };
    }

}
